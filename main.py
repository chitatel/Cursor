"""
RAG сервис v5.0 — production pipeline
- FAISS (IndexFlatIP) + LlamaIndex SentenceSplitter
- Полноценный hybrid retrieval: FAISS vector search + глобальный BM25
  по всему корпусу, объединение через RRF fusion
- BM25 с prefix-stemming (5-char prefix) для морфологических вариантов
  русского языка ("согласовать" ↔ "согласования")
- Document-aware context: соседние чанки того же документа подтягиваются
  автоматически, чтобы ответы по большим документам не строились на
  изолированных фрагментах
- Адаптивный chunking (cheatsheet 350/50 vs manual 800/120)
- Overlap-эвристика защиты от галлюцинаций (без доп. LLM-вызова)
- Батчевые эмбеддинги (×N быстрее индексирования)
- Thread-safe in-memory кэш записей; index.json с recovery при повреждении
- _faiss_index сбрасывается в None ДО записи JSON и rebuild — при сбое
  rebuild поиск вернёт пустой результат, а не данные от старого индекса
- index.json пишется раньше index.faiss; FAISS всегда перестраивается из JSON
  на старте — защита от рассинхронизации в том числе при совпадении ntotal
- Бэкапы битого JSON именуются с timestamp (index.json.bak.<ts>), не перезаписываются
- Path traversal защита на всех файловых эндпоинтах (Python 3.8+ совместимо)
- Потоковая загрузка файлов (без буферизации в памяти)
"""

import os
import re
import json
import math
import shutil
import logging
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import numpy as np
import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read config file '{CONFIG_PATH}': {e}") from e


CONFIG = _load_config()


def _config_value(name: str):
    if name not in CONFIG:
        raise RuntimeError(f"Missing required config key '{name}' in '{CONFIG_PATH}'")
    return CONFIG[name]


OLLAMA_BASE_URL = _config_value("OLLAMA_BASE_URL")
OLLAMA_LLM_MODEL = _config_value("OLLAMA_LLM_MODEL")
OLLAMA_EMBED_MODEL = _config_value("OLLAMA_EMBED_MODEL")
LLM_API_MODE = str(_config_value("LLM_API_MODE")).strip().lower()
STORAGE_DIR = Path(_config_value("STORAGE_DIR"))
if not STORAGE_DIR.is_absolute():
    STORAGE_DIR = (APP_DIR / STORAGE_DIR).resolve()
FILES_DIR = STORAGE_DIR / "files"
INDEX_FILE = STORAGE_DIR / "index.json"
FAISS_INDEX_FILE = STORAGE_DIR / "index.faiss"
TOP_K = int(_config_value("TOP_K"))
CHUNK_SIZE = int(_config_value("CHUNK_SIZE"))
CHUNK_OVERLAP = int(_config_value("CHUNK_OVERLAP"))
BASE_URL = CONFIG.get("BASE_URL", "")


def _public_base_url(request: Request) -> str:
    """Берём BASE_URL из конфига, а если пустой — из входящего запроса."""
    if BASE_URL:
        return BASE_URL.rstrip("/")
    return str(request.base_url).rstrip("/")
SIM_THRESHOLD = float(_config_value("SIM_THRESHOLD"))
OLLAMA_API_KEY = _config_value("OLLAMA_API_KEY")
OPENWEBUI_AUTH_PATH = _config_value("OPENWEBUI_AUTH_PATH")
OPENWEBUI_USER = _config_value("OPENWEBUI_USER")
OPENWEBUI_PASSWORD = _config_value("OPENWEBUI_PASSWORD")

# Размер батча для эмбеддингов при индексировании
EMBED_BATCH_SIZE = int(CONFIG.get("EMBED_BATCH_SIZE", 32))
# Включить query rewrite (добавляет 1 LLM-вызов, может улучшить recall)
QUERY_REWRITE_ENABLED = bool(CONFIG.get("QUERY_REWRITE_ENABLED", False))
# Вес BM25 при RRF fusion (0.0 — только вектор, 1.0 — только BM25)
# BM25 работает глобально по всему корпусу с prefix-stemming
BM25_WEIGHT = float(CONFIG.get("BM25_WEIGHT", 0.5))

os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STORAGE_DIR.mkdir(exist_ok=True)
FILES_DIR.mkdir(exist_ok=True)

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".msg"}

# ── Глобальное состояние ─────────────────────────────────────────────────────

_records: Optional[list[dict]] = None
_records_lock = asyncio.Lock()
_faiss_index = None
_indexing: dict[str, dict] = {}
_indexing_locks: dict[str, asyncio.Lock] = {}
_auth_token: Optional[str] = None
_auth_lock = asyncio.Lock()


# ── Хранилище записей ────────────────────────────────────────────────────────

def _load_records_sync() -> list[dict]:
    """
    Синхронное чтение с диска (вызывать только под _records_lock).
    При повреждённом JSON переименовывает файл в index.json.bak.<timestamp>
    и стартует с пустым индексом, чтобы сервис поднялся, а не упал.
    Timestamp в имени гарантирует, что предыдущие бэкапы не перезаписываются.
    """
    global _records
    if _records is None:
        if INDEX_FILE.exists():
            try:
                _records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                bak = INDEX_FILE.with_name(f"index.json.bak.{ts}")
                log.error(
                    "index.json is corrupted (%s). Renaming to %s and starting with empty index.",
                    e,
                    bak,
                )
                try:
                    INDEX_FILE.rename(bak)
                except Exception as rename_err:
                    log.error("Could not rename corrupted index.json: %s", rename_err)
                _records = []
        else:
            _records = []
    return _records


async def _get_records() -> list[dict]:
    """Потокобезопасное получение записей."""
    async with _records_lock:
        return list(_load_records_sync())


def _save_records_sync(records: list[dict]):
    """Атомарная запись на диск (вызывать только под _records_lock)."""
    global _records
    tmp = INDEX_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    tmp.replace(INDEX_FILE)
    _records = records


def _normalize_embedding(embedding: list[float]) -> list[float]:
    vec = np.asarray(embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ── FAISS ────────────────────────────────────────────────────────────────────

def _load_faiss():
    import faiss

    return faiss


def _rebuild_faiss_index_sync(records: list[dict]):
    """Перестроить FAISS индекс (вызывать только под _records_lock)."""
    global _faiss_index
    if not records:
        _faiss_index = None
        if FAISS_INDEX_FILE.exists():
            FAISS_INDEX_FILE.unlink()
        return

    faiss = _load_faiss()
    dim = len(records[0]["embedding"])
    index = faiss.IndexFlatIP(dim)
    embeddings = np.asarray([r["embedding"] for r in records], dtype="float32")
    index.add(embeddings)
    tmp = FAISS_INDEX_FILE.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(FAISS_INDEX_FILE)
    _faiss_index = index


def _get_faiss_index_sync(records: list[dict]):
    """Получить FAISS индекс (вызывать только под _records_lock)."""
    global _faiss_index
    if not records:
        _faiss_index = None
        return None
    if _faiss_index is None:
        faiss = _load_faiss()
        if FAISS_INDEX_FILE.exists():
            index = faiss.read_index(str(FAISS_INDEX_FILE))
            if index.ntotal != len(records):
                log.warning(
                    "FAISS index mismatch: ntotal=%s, records=%s. Rebuilding.",
                    index.ntotal,
                    len(records),
                )
                _rebuild_faiss_index_sync(records)
            else:
                first = np.asarray(records[0]["embedding"], dtype="float32")
                norm = float(np.linalg.norm(first))
                if abs(norm - 1.0) > 0.01:
                    log.warning("Embeddings not normalized (norm=%.3f). Rebuilding.", norm)
                    for r in records:
                        r["embedding"] = _normalize_embedding(r["embedding"])
                    _save_records_sync(records)
                    _rebuild_faiss_index_sync(records)
                else:
                    _faiss_index = index
        else:
            _rebuild_faiss_index_sync(records)
    return _faiss_index


# ── Операции над записями (потокобезопасные) ─────────────────────────────────

async def _chunk_count() -> int:
    return len(await _get_records())


async def _document_chunk_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in await _get_records():
        fn = record["filename"]
        counts[fn] = counts.get(fn, 0) + 1
    return counts


async def _document_records(filename: str) -> list[dict]:
    return [r for r in await _get_records() if r["filename"] == filename]


async def _replace_document_records(filename: str, new_records: list[dict]) -> int:
    global _faiss_index
    async with _records_lock:
        current = _load_records_sync()
        records = [r for r in current if r["filename"] != filename]
        removed = len(current) - len(records)
        records.extend(new_records)
        _faiss_index = None
        _save_records_sync(records)
        _rebuild_faiss_index_sync(records)
        return removed


async def _delete_document_records(filename: str) -> int:
    global _faiss_index
    async with _records_lock:
        current = _load_records_sync()
        records = [r for r in current if r["filename"] != filename]
        removed = len(current) - len(records)
        if removed == 0:
            return 0
        _faiss_index = None
        _save_records_sync(records)
        _rebuild_faiss_index_sync(records)
        return removed


# ── Безопасность файловых путей ───────────────────────────────────────────────

def _safe_filename(filename: str) -> str:
    """
    Проверяет, что имя файла не выходит за пределы FILES_DIR.
    Возбуждает HTTPException(400) при path traversal попытке.
    Возвращает нормализованное имя файла (только basename).
    Совместимо с Python 3.8+ (не использует Path.is_relative_to).
    """
    name = Path(filename).name
    if not name or name != filename:
        raise HTTPException(400, f"Invalid filename: '{filename}'")
    resolved = str((FILES_DIR / name).resolve())
    files_dir = str(FILES_DIR.resolve())
    if not resolved.startswith(files_dir + os.sep) and resolved != files_dir:
        raise HTTPException(400, f"Invalid filename: '{filename}'")
    return name


def _bm25_scores(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    BM25 с prefix-stemming: для токенов ≥5 символов при отсутствии точного
    совпадения проверяется совпадение первых 5 символов.  Это покрывает
    основные морфологические вариации русского языка без внешнего стеммера.
    """
    _STEM_LEN = 5
    tokenize = lambda t: re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", t.lower())
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return [0.0] * len(texts)

    tokenized = [tokenize(t) for t in texts]

    # Precompute per-document: exact token set + prefix set
    token_sets = [set(toks) for toks in tokenized]
    prefix_sets = [
        set(tok[:_STEM_LEN] for tok in toks if len(tok) >= _STEM_LEN)
        for toks in tokenized
    ]

    # Precompute DF for each query term (exact OR prefix match)
    df_cache: dict[str, int] = {}
    for term in q_tokens:
        df = 0
        term_prefix = term[:_STEM_LEN] if len(term) >= _STEM_LEN else None
        for i in range(len(tokenized)):
            if term in token_sets[i]:
                df += 1
            elif term_prefix and term_prefix in prefix_sets[i]:
                df += 1
        df_cache[term] = df

    doc_lens = [len(t) for t in tokenized]
    avg_dl = sum(doc_lens) / max(len(doc_lens), 1)
    n_docs = len(texts)

    scores = []
    for tokens in tokenized:
        tf_map: dict[str, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        prefix_tf: dict[str, int] = {}
        for tok in tokens:
            if len(tok) >= _STEM_LEN:
                p = tok[:_STEM_LEN]
                prefix_tf[p] = prefix_tf.get(p, 0) + 1

        score = 0.0
        dl = len(tokens)
        for term in q_tokens:
            df = df_cache[term]
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            tf = tf_map.get(term, 0)
            if tf == 0 and len(term) >= _STEM_LEN:
                tf = prefix_tf.get(term[:_STEM_LEN], 0)
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm
        scores.append(score)
    return scores


def _filename_match_boost(query: str, filename: str) -> float:
    """
    Небольшой лексический boost по имени файла.
    Это помогает общим запросам вроде "как согласовать документ" поднимать
    файлы, в названии которых есть близкие ключи, не ломая основной ranking.
    """
    query_tokens = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", query.lower()))
    if not query_tokens:
        return 0.0
    filename_tokens = set(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", Path(filename).stem.lower()))
    if not filename_tokens:
        return 0.0
    overlap = len(query_tokens & filename_tokens)
    # Boost небольшой и дискретный: он только помогает shortlist, а не заменяет FAISS.
    return min(overlap * 0.35, 1.4)


def _rrf_fusion(
    vector_ids: list[str],
    bm25_ids: list[str],
    bm25_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[str]:
    """
    Reciprocal Rank Fusion двух ранжированных списков.
    Возвращает список id, отсортированных по убыванию RRF-score.
    """
    scores: dict[str, float] = {}
    vector_weight = 1.0 - bm25_weight

    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (rrf_k + rank + 1)

    return sorted(scores, key=lambda x: scores[x], reverse=True)


async def _search_records(query: str, query_embedding: list[float], top_k: int) -> dict:
    """Гибридный поиск: FAISS-кандидаты + глобальные lexical-кандидаты + RRF fusion."""
    async with _records_lock:
        records = _load_records_sync()
        index = _get_faiss_index_sync(records)

    if not records:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    id_to_record = {r["id"]: r for r in records}
    vector_ids: list[str] = []
    vector_distances: dict[str, float] = {}
    if index is not None:
        vec = np.asarray([query_embedding], dtype="float32")
        vector_fetch_k = min(max(top_k * 8, 12), len(records))
        scores, indices = index.search(vec, vector_fetch_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = records[int(idx)]
            vector_ids.append(record["id"])
            vector_distances[record["id"]] = float(1 - score)

    lexical_fetch_k = min(max(top_k * 8, 12), len(records))
    lexical_texts = [f"{record['filename']} {record['text']}" for record in records]
    lexical_scores = _bm25_scores(query, lexical_texts)
    lexical_scores = [
        score + _filename_match_boost(query, record["filename"])
        for score, record in zip(lexical_scores, records)
    ]
    lexical_ids = [
        records[i]["id"]
        for i in sorted(range(len(lexical_scores)), key=lambda x: lexical_scores[x], reverse=True)[:lexical_fetch_k]
        if lexical_scores[i] > 0
    ]

    fused_ids = _rrf_fusion(vector_ids, lexical_ids, bm25_weight=BM25_WEIGHT)[:top_k]

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    distances: list[float] = []
    for doc_id in fused_ids:
        if doc_id not in id_to_record:
            continue
        record = id_to_record[doc_id]
        ids.append(record["id"])
        docs.append(record["text"])
        metas.append({
            "filename": record["filename"],
            "chunk_index": record.get("chunk_index", 0),
        })
        distances.append(vector_distances.get(doc_id, 1.0))

    return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [distances]}


def _build_document_aware_context(
    result_ids: list[str],
    records: list[dict],
    budget: int,
) -> tuple[list[str], list[dict]]:
    """
    Расширяет найденные чанки соседними чанками того же документа (±1).
    Это критично для больших документов: если вопрос затрагивает процесс
    из нескольких шагов, ответ не должен строиться на изолированных фрагментах.

    Документы обрабатываются в порядке ранжирования — наиболее релевантный
    документ получает приоритет по бюджету.  Чанки внутри документа
    добавляются в порядке chunk_index (естественный порядок текста).

    budget — максимальное число чанков в итоговом контексте.
    """
    id_to_record = {r["id"]: r for r in records}

    # Сгруппировать выбранные чанки по документу (с сохранением порядка ранжирования)
    doc_selected: dict[str, list[int]] = {}
    doc_order: list[str] = []
    for rid in result_ids:
        rec = id_to_record.get(rid)
        if not rec:
            continue
        fn = rec["filename"]
        ci = rec.get("chunk_index", 0)
        if fn not in doc_selected:
            doc_selected[fn] = []
            doc_order.append(fn)
        doc_selected[fn].append(ci)

    # Построить карту чанков для каждого релевантного документа
    doc_chunks: dict[str, dict[int, dict]] = {}
    for rec in records:
        fn = rec["filename"]
        if fn not in doc_selected:
            continue
        ci = rec.get("chunk_index", 0)
        doc_chunks.setdefault(fn, {})[ci] = rec

    used = 0
    context_texts: list[str] = []
    context_metas: list[dict] = []

    for fn in doc_order:
        if used >= budget:
            break

        chunks_map = doc_chunks.get(fn, {})
        if not chunks_map:
            continue
        max_ci = max(chunks_map.keys())
        selected = sorted(doc_selected[fn])

        # Расширить окно: ±1 сосед для каждого выбранного чанка
        expanded = set()
        for ci in selected:
            for delta in range(-1, 2):
                neighbor = ci + delta
                if 0 <= neighbor <= max_ci and neighbor in chunks_map:
                    expanded.add(neighbor)

        # Добавить чанки в порядке их следования в документе
        for ci in sorted(expanded):
            if used >= budget:
                break
            rec = chunks_map[ci]
            context_texts.append(rec["text"])
            context_metas.append({"filename": fn, "chunk_index": ci})
            used += 1

    return context_texts, context_metas


# ── Auth / HTTP ───────────────────────────────────────────────────────────────

async def _get_auth_token(force_refresh: bool = False) -> str:
    global _auth_token

    if OLLAMA_API_KEY:
        return OLLAMA_API_KEY

    if not OPENWEBUI_PASSWORD or not OPENWEBUI_USER:
        return ""

    if _auth_token and not force_refresh:
        return _auth_token

    async with _auth_lock:
        if _auth_token and not force_refresh:
            return _auth_token

        signin_url = f"{_openwebui_root_url()}{OPENWEBUI_AUTH_PATH}"
        payload = {"user": OPENWEBUI_USER, "password": OPENWEBUI_PASSWORD}

        async with httpx.AsyncClient(timeout=60) as c:
            response = await c.post(
                signin_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

        token = (
            data.get("token")
            or data.get("access_token")
            or data.get("data", {}).get("token")
        )
        if not token and "token=" in response.headers.get("set-cookie", ""):
            cookie = response.headers["set-cookie"].split("token=", 1)[1]
            token = cookie.split(";", 1)[0]
        if not token:
            raise ValueError("Open WebUI signin succeeded but token was not returned")

        _auth_token = token
        return token


async def _api_headers(force_refresh: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    token = await _get_auth_token(force_refresh=force_refresh)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _base_url() -> str:
    return OLLAMA_BASE_URL.rstrip("/")


def _api_mode() -> str:
    if LLM_API_MODE in {"ollama", "openai", "openwebui"}:
        return LLM_API_MODE
    base = _base_url().lower()
    if "/api/chat/completions" in base or "/ollama/api" in base:
        return "openwebui"
    if base.endswith("/v1") or "/api/v1" in base:
        return "openai"
    if OLLAMA_API_KEY:
        return "openwebui"
    return "ollama"


def _openai_base_url() -> str:
    base = _base_url()
    if base.endswith("/v1") or base.endswith("/api/v1"):
        return base
    return f"{base}/api/v1"


def _openwebui_root_url() -> str:
    base = _base_url()
    for suffix in ("/api/chat/completions", "/ollama/api/embed", "/ollama/api", "/api"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _chat_url() -> str:
    mode = _api_mode()
    if mode == "openai":
        return f"{_openai_base_url()}/chat/completions"
    if mode == "openwebui":
        return f"{_openwebui_root_url()}/api/chat/completions"
    return f"{_base_url()}/api/chat"


def _embed_url() -> str:
    mode = _api_mode()
    if mode == "openai":
        return f"{_openai_base_url()}/embeddings"
    if mode == "openwebui":
        return f"{_openwebui_root_url()}/ollama/api/embed"
    return f"{_base_url()}/api/embed"


# ── Embed (с батчингом) ───────────────────────────────────────────────────────

async def _embed_one(text: str, client: httpx.AsyncClient) -> Optional[list[float]]:
    headers = await _api_headers()
    response = await client.post(
        _embed_url(),
        headers=headers,
        json={"model": OLLAMA_EMBED_MODEL, "input": text},
    )
    if response.status_code == 401 and OPENWEBUI_PASSWORD and OPENWEBUI_USER:
        headers = await _api_headers(force_refresh=True)
        response = await client.post(
            _embed_url(),
            headers=headers,
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
        )
    if response.status_code == 400:
        return None
    response.raise_for_status()
    data = response.json()
    if "data" in data:
        return data["data"][0]["embedding"]
    embeddings = data["embeddings"]
    return embeddings[0] if isinstance(embeddings[0], list) else embeddings


async def _embed(text: str) -> Optional[list[float]]:
    """Получить эмбеддинг для одного текста."""
    async with httpx.AsyncClient(timeout=120) as c:
        return await _embed_one(text, c)


async def _embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Батчевые эмбеддинги: один HTTP-запрос на батч (если модель поддерживает),
    с fallback на поштучный запрос при ошибке.
    """
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=300) as c:
        headers = await _api_headers()
        try:
            response = await c.post(
                _embed_url(),
                headers=headers,
                json={"model": OLLAMA_EMBED_MODEL, "input": texts},
            )
            if response.status_code == 401 and OPENWEBUI_PASSWORD and OPENWEBUI_USER:
                headers = await _api_headers(force_refresh=True)
                response = await c.post(
                    _embed_url(),
                    headers=headers,
                    json={"model": OLLAMA_EMBED_MODEL, "input": texts},
                )

            if response.status_code == 400:
                raise ValueError("batch not supported")

            response.raise_for_status()
            data = response.json()

            if "data" in data:
                sorted_items = sorted(data["data"], key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in sorted_items]

            embeddings = data.get("embeddings", [])
            if embeddings and isinstance(embeddings[0], list):
                return embeddings
            raise ValueError("unexpected batch response shape")

        except Exception as e:
            log.warning("Batch embed failed (%s), falling back to sequential", e)
            results = []
            for text in texts:
                results.append(await _embed_one(text, c))
            return results


# ── LLM chat ─────────────────────────────────────────────────────────────────

async def _chat(messages: list[dict], max_tokens: int = 400) -> str:
    async with httpx.AsyncClient(timeout=600) as c:
        headers = await _api_headers()
        mode = _api_mode()

        payload_openai = {
            "model": OLLAMA_LLM_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
            "top_p": 0.85,
            "max_tokens": max_tokens,
        }
        payload_ollama = {
            "model": OLLAMA_LLM_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.85,
                "top_k": 30,
                "repeat_penalty": 1.15,
                "num_ctx": 8192,
            },
        }

        async def _post(payload):
            response = await c.post(_chat_url(), headers=headers, json=payload)
            if response.status_code == 401 and OPENWEBUI_PASSWORD and OPENWEBUI_USER:
                refreshed_headers = await _api_headers(force_refresh=True)
                response = await c.post(_chat_url(), headers=refreshed_headers, json=payload)
            response.raise_for_status()
            return response

        if mode in {"openai", "openwebui"}:
            response = await _post(payload_openai)
            content = response.json()["choices"][0]["message"]["content"]
        else:
            response = await _post(payload_ollama)
            content = response.json().get("message", {}).get("content", "")

        if not content:
            raise ValueError("LLM вернул пустой ответ")
        return content


# ── Query rewrite ─────────────────────────────────────────────────────────────

async def _rewrite_query(q: str) -> str:
    """
    Query rewrite отключён.  Морфологические вариации покрываются
    prefix-stemming в BM25, а семантические — FAISS.  Дополнительный
    LLM-вызов на rewrite добавляет latency без гарантированного выигрыша
    по recall и может подменять смысл вопроса.
    """
    return q


# ── Faithfulness check ────────────────────────────────────────────────────────

def _is_faithful(answer: str, context: str) -> bool:
    """
    Быстрая лексическая проверка: ответ должен опираться на лексику контекста.
    Работает без LLM-вызова — не увеличивает латентность.
    Контекст не обрезается: проверяем по всему тексту, а не первым 3000 символам.
    """

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    ctx_norm = normalize(context)
    ans_norm = normalize(answer)
    words = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{4,}", ans_norm)]
    if not words:
        return True
    overlap = sum(1 for w in words if w in ctx_norm)
    ratio = overlap / len(words)
    if ratio < 0.3:
        log.warning(
            "Faithfulness check: low overlap %.0f%% (%d/%d words)",
            ratio * 100,
            overlap,
            len(words),
        )
        return False
    return True


# ── Форматирование ответа в HTML ─────────────────────────────────────────────

def _answer_to_html(
    answer: str,
    image_urls: dict[str, str],
    download_urls: dict[str, str] | None = None,
) -> str:
    """
    Конвертирует текстовый ответ в HTML для отображения в 1С.
    - Нумерованные пункты → <ol><li>
    - URL картинок → <img> теги
    - Ссылки на документы-источники внизу
    - Остальной текст → <p>
    """
    from html import escape

    lines = answer.split("\n")
    html_parts: list[str] = []
    in_list = False
    numbered_re = re.compile(r"^(\d+)[\.\)]\s+(.*)")

    url_set = set(image_urls.values()) if image_urls else set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Строка — это URL картинки
        if stripped in url_set:
            if in_list:
                html_parts.append(
                    f'<br><img src="{escape(stripped)}" '
                    f'style="max-width:100%; margin-top:8px; border:1px solid #ccc; border-radius:4px;">'
                )
            else:
                html_parts.append(
                    f'<p><img src="{escape(stripped)}" '
                    f'style="max-width:100%; border:1px solid #ccc; border-radius:4px;"></p>'
                )
            continue

        m = numbered_re.match(stripped)
        if m:
            if not in_list:
                html_parts.append("<ol>")
                in_list = True
            html_parts.append(f"<li>{escape(m.group(2))}")
            continue

        # Обычная строка
        if in_list:
            html_parts.append("</li></ol>")
            in_list = False
        html_parts.append(f"<p>{escape(stripped)}</p>")

    if in_list:
        html_parts.append("</li></ol>")

    # Блок ссылок на документы-источники
    if download_urls:
        html_parts.append('<hr style="margin-top:16px; border:none; border-top:1px solid #ddd;">')
        html_parts.append('<p style="margin-top:12px; font-size:13px; color:#666;">Источники:</p>')
        html_parts.append('<ul style="padding-left:20px; font-size:13px;">')
        for filename, url in download_urls.items():
            html_parts.append(
                f'<li><a href="{escape(url)}" style="color:#1a73e8; text-decoration:none;">'
                f'{escape(filename)}</a></li>'
            )
        html_parts.append("</ul>")

    body = "\n".join(html_parts)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<style>"
        "body { font-family: Arial, sans-serif; font-size: 14px; padding: 12px; color: #333; }"
        "ol { padding-left: 20px; }"
        "li { margin-bottom: 10px; }"
        "p { margin: 8px 0; }"
        "img { display: block; }"
        "a:hover { text-decoration: underline !important; }"
        "</style>"
        f"</head><body>{body}</body></html>"
    )


# ── Привязка иллюстраций к пунктам ответа ────────────────────────────────────

def _attach_images_to_answer(
    answer: str,
    chunks: list[str],
    chunk_metas: list[dict],
    image_urls: dict[str, str],
) -> str:
    """
    Группируем картинки, идущие подряд в чанке (без значимого текста между ними).
    Для каждой группы берём текст перед первой картинкой группы.
    Для каждого пункта ответа ищем группу с наибольшим пересечением слов.
    Привязываем всю группу (все картинки) к пункту.
    """
    img_marker_re = re.compile(r"\[Рисунок \d+: [^\]]+\]")

    def bag(text: str) -> set[str]:
        cleaned = img_marker_re.sub("", text)
        return set(re.findall(r"[А-Яа-яЁёA-Za-z0-9-]{4,}", cleaned.lower()))

    # Группируем маркеры: если между двумя маркерами нет значимого текста
    # (менее 3 слов), они образуют одну группу
    # group = (preceding_bag, [url1, url2, ...])
    groups: list[tuple[set[str], list[str]]] = []
    for chunk_text in chunks:
        markers_in_chunk = list(img_marker_re.finditer(chunk_text))
        if not markers_in_chunk:
            continue

        current_group_urls: list[str] = []
        current_group_preceding: set[str] = set()

        for i, match in enumerate(markers_in_chunk):
            url = image_urls.get(match.group(0))
            if not url:
                continue

            start = markers_in_chunk[i - 1].end() if i > 0 else 0
            preceding_text = chunk_text[start:match.start()]
            preceding_words = bag(preceding_text)

            if i == 0 or len(preceding_words) >= 1:
                # Начало новой группы: первый маркер или есть хоть одно слово перед ним
                if current_group_urls:
                    groups.append((current_group_preceding, current_group_urls))
                current_group_preceding = preceding_words
                current_group_urls = [url]
            else:
                # Продолжение группы: между маркерами мало текста
                current_group_urls.append(url)

        if current_group_urls:
            groups.append((current_group_preceding, current_group_urls))

    if not groups:
        return answer

    # Разбиваем ответ на пункты
    point_re = re.compile(r"(?:^|\n)(\d+[\.\)]\s+.*?)(?=\n\d+[\.\)]\s+|\Z)", re.DOTALL)
    points = point_re.findall(answer)
    if not points:
        points = [line for line in answer.split("\n") if line.strip()]
    if not points:
        return answer

    used_groups: set[int] = set()
    result_parts: list[str] = []

    for point in points:
        point_bag = bag(point)
        if not point_bag:
            result_parts.append(point)
            continue

        best_idx = -1
        best_score = 0
        for idx, (preceding_bag, urls) in enumerate(groups):
            if idx in used_groups:
                continue
            score = len(point_bag & preceding_bag)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= 2:
            used_groups.add(best_idx)
            all_urls = "\n".join(groups[best_idx][1])
            result_parts.append(f"{point}\n{all_urls}")
        else:
            result_parts.append(point)

    return "\n".join(result_parts)


# ── Извлечение изображений из документов ─────────────────────────────────────

def _extract_docx_with_images(path: Path) -> list:
    """
    Извлекает текст из .docx с сохранением изображений.
    Изображения сохраняются в FILES_DIR/<stem>_images/img_001.png и т.д.
    В текст вставляются маркеры [Рисунок N: img_NNN.ext] на месте картинок.
    """
    from docx import Document as DocxDocument
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from llama_index.core.schema import Document

    doc = DocxDocument(str(path))
    stem = path.stem
    images_dir = FILES_DIR / f"{stem}_images"
    images_dir.mkdir(exist_ok=True)

    # Собираем все inline-изображения с привязкой к параграфам
    img_counter = 0
    para_images: dict[int, list[str]] = {}  # para_index -> list of markers

    for para_idx, para in enumerate(doc.paragraphs):
        for run in para.runs:
            drawing_elements = run._element.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"
            ) + run._element.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pict"
            )
            # Inline images через drawing
            blips = run._element.findall(
                ".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
            )
            for blip in blips:
                embed_id = blip.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                )
                if not embed_id:
                    continue
                try:
                    rel = doc.part.rels[embed_id]
                except KeyError:
                    continue
                img_counter += 1
                img_data = rel.target_part.blob
                content_type = rel.target_part.content_type
                ext_map = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/gif": ".gif",
                    "image/bmp": ".bmp",
                    "image/tiff": ".tiff",
                    "image/x-wmf": ".wmf",
                    "image/x-emf": ".emf",
                }
                ext = ext_map.get(content_type, ".png")
                img_name = f"img_{img_counter:03d}{ext}"
                img_path = images_dir / img_name
                img_path.write_bytes(img_data)
                marker = f"[Рисунок {img_counter}: {img_name}]"
                para_images.setdefault(para_idx, []).append(marker)
                log.info("[%s] Extracted image: %s", path.name, img_name)

    # Собираем текст с маркерами
    parts = []
    for para_idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            parts.append(text)
        if para_idx in para_images:
            for marker in para_images[para_idx]:
                parts.append(marker)

    full_text = "\n".join(parts).strip()
    if not full_text:
        raise ValueError("No text extracted from DOCX file")

    # Удаляем папку, если картинок не нашлось
    if img_counter == 0:
        images_dir.rmdir()

    log.info("[%s] Extracted %d images", path.name, img_counter)
    return [Document(text=full_text, metadata={"filename": path.name})]


# ── Загрузка документов ───────────────────────────────────────────────────────

def _load_documents(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return _extract_docx_with_images(path)

    if suffix == ".msg":
        try:
            import extract_msg
            from llama_index.core.schema import Document
        except ImportError as e:
            raise RuntimeError("MSG support requires 'extract-msg' package") from e

        message = extract_msg.Message(str(path))
        parts = []
        subject = (message.subject or "").strip()
        sender = (message.sender or "").strip()
        date = (message.date or "").strip()
        body = (message.body or "").strip()
        if subject:
            parts.append(f"Subject: {subject}")
        if sender:
            parts.append(f"From: {sender}")
        if date:
            parts.append(f"Date: {date}")
        if body:
            parts.append("")
            parts.append(body)
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("No text extracted from MSG file")
        return [Document(text=text, metadata={"filename": path.name})]

    from llama_index.core import SimpleDirectoryReader

    return SimpleDirectoryReader(input_files=[str(path)]).load_data()


def _detect_chunk_profile(texts: list[str]) -> tuple[str, int, int]:
    """
    Простейшая эвристика для mixed corpus:
    - cheatsheet: много коротких строк, списков и коротких заголовков
    - manual: более длинные связные абзацы
    """
    merged = "\n".join(t for t in texts if t).strip()
    if not merged:
        return ("manual", CHUNK_SIZE, CHUNK_OVERLAP)

    lines = [line.strip() for line in merged.splitlines() if line.strip()]
    if not lines:
        return ("manual", CHUNK_SIZE, CHUNK_OVERLAP)

    short_lines = sum(1 for line in lines if len(line) <= 80)
    numbered_lines = sum(1 for line in lines if re.match(r"^\d+[\.\)]\s+", line))
    heading_like = sum(
        1
        for line in lines
        if len(line) <= 120 and not line.endswith(".") and len(line.split()) <= 12
    )

    total_lines = len(lines)
    short_ratio = short_lines / total_lines
    numbered_ratio = numbered_lines / total_lines
    heading_ratio = heading_like / total_lines

    if short_ratio > 0.45 or numbered_ratio > 0.15 or heading_ratio > 0.20:
        return ("cheatsheet", 350, 50)
    return ("manual", 800, 120)


# ── Индексирование ────────────────────────────────────────────────────────────

async def _run_indexing(filename: str, dest: Path):
    lock = asyncio.Lock()
    _indexing_locks[filename] = lock

    async with lock:
        _indexing[filename] = {
            "status": "indexing",
            "chunks_done": 0,
            "chunks_total": 0,
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
        try:
            from llama_index.core.node_parser import SentenceSplitter

            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, lambda: _load_documents(dest))
            if not docs:
                raise ValueError("No text extracted from file")

            doc_texts = [getattr(doc, "text", "").strip() for doc in docs if getattr(doc, "text", "").strip()]
            profile, effective_chunk_size, effective_chunk_overlap = _detect_chunk_profile(doc_texts)
            log.info(
                "[%s] chunk profile: %s (%d/%d)",
                filename,
                profile,
                effective_chunk_size,
                effective_chunk_overlap,
            )

            splitter = SentenceSplitter(
                chunk_size=effective_chunk_size,
                chunk_overlap=effective_chunk_overlap,
            )
            nodes = await loop.run_in_executor(
                None,
                lambda: splitter.get_nodes_from_documents(docs),
            )

            texts = [node.text.strip() for node in nodes if node.text.strip()]
            total = len(texts)
            _indexing[filename]["chunks_total"] = total
            log.info("[%s] %d chunks (SentenceSplitter)", filename, total)

            done = 0
            new_records: list[dict] = []

            for batch_start in range(0, total, EMBED_BATCH_SIZE):
                batch_texts = texts[batch_start: batch_start + EMBED_BATCH_SIZE]
                embeddings = await _embed_batch(batch_texts)

                for offset, (text, emb) in enumerate(zip(batch_texts, embeddings)):
                    if emb is None:
                        log.warning("[%s] skipping chunk %d (embed 400)", filename, batch_start + offset)
                        continue
                    new_records.append(
                        {
                            "id": f"{filename}_{batch_start + offset}",
                            "filename": filename,
                            "chunk_index": batch_start + offset,
                            "text": text,
                            "embedding": _normalize_embedding(emb),
                        }
                    )
                    done += 1

                _indexing[filename]["chunks_done"] = done
                log.info("[%s] embedded %d/%d", filename, done, total)

            await _replace_document_records(filename, new_records)

            _indexing[filename].update(
                {
                    "status": "ready",
                    "chunks_done": done,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            log.info("[%s] done: %d chunks", filename, done)

        except Exception as e:
            log.error("[%s] failed: %s", filename, e)
            _indexing[filename].update(
                {
                    "status": "error",
                    "error": str(e),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        finally:
            _indexing_locks.pop(filename, None)


# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    При старте всегда перестраиваем FAISS из JSON — это дешевле, чем
    доверять файлу на диске, который мог рассинхронизироваться (в том числе
    при совпадении ntotal, но изменённом содержимом после сбоя).
    """
    global _faiss_index
    async with _records_lock:
        records = _load_records_sync()
        if records:
            log.info("Rebuilding FAISS index from JSON (%d records)...", len(records))
            _rebuild_faiss_index_sync(records)
        else:
            _faiss_index = None
            log.info("Index is empty, skipping FAISS build.")
    log.info("FAISS index ready: %d chunks", len(records))
    yield


app = FastAPI(title="RAG Ollama API v5.0", version="5.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    answer_html: str
    sources: list[str]
    chunks_used: int
    download_urls: dict[str, str]
    image_urls: dict[str, str]
    raw_chunks: list[str]
    rewritten_query: Optional[str] = None


class UploadResponse(BaseModel):
    filename: str
    status: str


class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    indexing_status: str
    download_url: str


class IndexingStatus(BaseModel):
    filename: str
    status: str
    chunks_done: int
    chunks_total: int
    progress_pct: float
    error: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]


class StatusResponse(BaseModel):
    total_chunks: int
    total_documents: int
    llm_model: str
    embed_model: str
    ollama_url: str


@app.get("/status", response_model=StatusResponse)
async def status():
    total = await _chunk_count()
    docs = len(await _document_chunk_counts())
    return StatusResponse(
        total_chunks=total,
        total_documents=docs,
        llm_model=OLLAMA_LLM_MODEL,
        embed_model=OLLAMA_EMBED_MODEL,
        ollama_url=OLLAMA_BASE_URL,
    )


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents(request: Request):
    counts = await _document_chunk_counts()
    for fn in _indexing:
        counts.setdefault(fn, 0)
    base = _public_base_url(request)
    result = []
    for fn, n in sorted(counts.items()):
        st = _indexing.get(fn, {}).get("status", "ready" if n > 0 else "unknown")
        result.append(
            DocumentInfo(
                filename=fn,
                chunks=n,
                indexing_status=st,
                download_url=f"{base}/documents/{fn}/download",
            )
        )
    return result


@app.post("/documents", response_model=UploadResponse, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = _safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            400,
            f"Unsupported type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED))}",
        )
    if _indexing.get(filename, {}).get("status") == "indexing":
        raise HTTPException(409, f"'{filename}' is already being indexed")
    dest = FILES_DIR / filename
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 256):
            await f.write(chunk)
    background_tasks.add_task(_run_indexing, filename, dest)
    return UploadResponse(filename=filename, status="indexing_started")


@app.get("/documents/{filename}/download")
async def download_document(filename: str):
    filename = _safe_filename(filename)
    path = FILES_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"File '{filename}' not found")
    return FileResponse(path, filename=filename)


@app.get("/documents/{filename}/images/{image_name}")
async def download_image(filename: str, image_name: str):
    filename = _safe_filename(filename)
    image_name = Path(image_name).name
    if not image_name:
        raise HTTPException(400, "Invalid image name")
    stem = Path(filename).stem
    img_path = FILES_DIR / f"{stem}_images" / image_name
    if not img_path.exists():
        raise HTTPException(404, f"Image '{image_name}' not found for '{filename}'")
    return FileResponse(img_path, filename=image_name)


@app.get("/documents/{filename}/status", response_model=IndexingStatus)
async def document_status(filename: str):
    filename = _safe_filename(filename)
    info = _indexing.get(filename)
    if info is None:
        chunks = len(await _document_records(filename))
        if chunks:
            return IndexingStatus(
                filename=filename,
                status="ready",
                chunks_done=chunks,
                chunks_total=chunks,
                progress_pct=100.0,
                error=None,
                started_at=None,
                finished_at=None,
            )
        raise HTTPException(404, f"Document '{filename}' not found")
    total = info["chunks_total"] or 1
    return IndexingStatus(
        filename=filename,
        status=info["status"],
        chunks_done=info["chunks_done"],
        chunks_total=info["chunks_total"],
        progress_pct=round(info["chunks_done"] / total * 100, 1),
        error=info.get("error"),
        started_at=info.get("started_at"),
        finished_at=info.get("finished_at"),
    )


@app.post("/documents/{filename}/reindex")
async def reindex_document(filename: str, background_tasks: BackgroundTasks):
    """Переиндексировать существующий документ без повторной загрузки."""
    filename = _safe_filename(filename)
    dest = FILES_DIR / filename
    if not dest.exists():
        raise HTTPException(404, f"File '{filename}' not found")
    if _indexing.get(filename, {}).get("status") == "indexing":
        raise HTTPException(409, f"'{filename}' is already being indexed")
    # Удаляем старые чанки и картинки
    await _delete_document_records(filename)
    images_dir = FILES_DIR / f"{Path(filename).stem}_images"
    if images_dir.is_dir():
        shutil.rmtree(images_dir, ignore_errors=True)
    background_tasks.add_task(_run_indexing, filename, dest)
    return {"filename": filename, "status": "reindexing_started"}


@app.post("/documents/reindex-all")
async def reindex_all(background_tasks: BackgroundTasks):
    """Переиндексировать все документы."""
    files = [f for f in FILES_DIR.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED]
    if not files:
        raise HTTPException(404, "No documents found")
    started = []
    skipped = []
    for dest in files:
        fn = dest.name
        if _indexing.get(fn, {}).get("status") == "indexing":
            skipped.append(fn)
            continue
        await _delete_document_records(fn)
        images_dir = FILES_DIR / f"{dest.stem}_images"
        if images_dir.is_dir():
            shutil.rmtree(images_dir, ignore_errors=True)
        background_tasks.add_task(_run_indexing, fn, dest)
        started.append(fn)
    return {"started": started, "skipped_already_indexing": skipped}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    filename = _safe_filename(filename)
    if _indexing.get(filename, {}).get("status") == "indexing":
        raise HTTPException(409, f"'{filename}' is currently being indexed")
    removed = await _delete_document_records(filename)
    if not removed:
        raise HTTPException(404, f"Document '{filename}' not found in index")
    (FILES_DIR / filename).unlink(missing_ok=True)
    # Удаляем папку с извлечёнными изображениями
    images_dir = FILES_DIR / f"{Path(filename).stem}_images"
    if images_dir.is_dir():
        shutil.rmtree(images_dir, ignore_errors=True)
    _indexing.pop(filename, None)
    return {"filename": filename, "chunks_removed": removed}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    if await _chunk_count() == 0:
        raise HTTPException(503, "Index is empty. Upload documents first via POST /documents")

    raw_top_k = req.top_k
    if raw_top_k is not None and (not isinstance(raw_top_k, int) or raw_top_k < 1):
        raise HTTPException(400, f"top_k must be a positive integer, got: {raw_top_k}")
    top_k = max(1, raw_top_k or TOP_K)

    rewritten = await _rewrite_query(req.question)
    log.info("Query rewrite: %r → %r", req.question, rewritten)

    try:
        q_emb = await _embed(rewritten)
        if q_emb is None:
            raise ValueError("embed returned None")
    except Exception as e:
        raise HTTPException(502, f"Embedding API error: {e}")

    results = await _search_records(rewritten, _normalize_embedding(q_emb), top_k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    _not_found = "Информация в документах не найдена."
    _not_found_html = _answer_to_html(_not_found, {})

    if not docs:
        return AskResponse(
            answer=_not_found,
            answer_html=_not_found_html,
            sources=[],
            chunks_used=0,
            download_urls={},
            image_urls={},
            raw_chunks=[],
            rewritten_query=rewritten,
        )

    best_sim = 1 - distances[0]
    log.info("Best similarity: %.3f", best_sim)
    if best_sim < SIM_THRESHOLD:
        return AskResponse(
            answer=_not_found,
            answer_html=_not_found_html,
            sources=[],
            chunks_used=0,
            download_urls={},
            image_urls={},
            raw_chunks=[],
            rewritten_query=rewritten,
        )

    # Расширяем контекст соседними чанками (document-aware)
    records = await _get_records()
    all_docs, all_metas = _build_document_aware_context(
        results["ids"][0], records, budget=top_k * 2,
    )

    if not all_docs:
        all_docs = docs
        all_metas = metas

    context = "\n\n".join(
        f"[CHUNK {i} | {m['filename']}]\n{d}"
        for i, (d, m) in enumerate(zip(all_docs, all_metas))
    )
    _all_sources = list(dict.fromkeys(m["filename"] for m in all_metas))

    try:
        answer = await _chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Отвечай только на русском языке.\n"
                        "Контекст ниже — единственный источник правды.\n"
                        "Строго следуй порядку изложения в контексте — не переставляй шаги местами и не начинай с середины.\n"
                        "Строго запрещено придумывать шаги, роли, ограничения, числа, лимиты, сроки и требования, которых нет в контексте.\n"
                        "Строго запрещено объединять информацию из разных разделов или подпунктов в один шаг.\n"
                        "Не пропускай разделы из контекста — включай все: основные шаги, исправления ошибок, примечания, подсказки.\n"
                        "Если в контексте нет прямого ответа на вопрос, ответь ровно одной фразой: Информация в документах не найдена.\n"
                        "Оформляй ответ единым нумерованным списком со сквозной нумерацией (1. 2. 3. ...), каждый шаг — отдельным пунктом. Не перезапускай нумерацию.\n"
                        "Если спрашивают 'какие бывают', 'виды', 'типы', 'перечислить' — дай нумерованный список элементов; если спрашивают 'как сделать', 'порядок', 'процесс' — дай пошаговую инструкцию. Включай все значимые шаги из контекста, в том числе подсказки, исправления ошибок и важные замечания.\n"
                        "Не пиши фразы вроде: 'Давайте разберем', 'Вот пошаговая инструкция', 'Based on the provided documentation', 'Okay'.\n"
                        "Не переводи термины на английский и не смешивай языки.\n"
                        "Если есть сомнение, лучше ответь: Информация в документах не найдена."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Контекст:\n\n{context}\n\nВопрос: {req.question}",
                },
            ],
            max_tokens=1500,
        )
    except Exception as e:
        raise HTTPException(502, f"Chat API error: {e}")

    log.info("Raw answer: %r", answer[:200])

    # Убираем блок рассуждений <think>...</think> (deepseek-r1 и подобные)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    # Убираем маркеры чанков [CHUNK N] / (CHUNK N), которые LLM может скопировать из контекста
    answer = re.sub(r"\s*[\(\[]\s*CHUNK\s*\d+\s*[\)\]]", "", answer).strip()

    answer_clean = answer.strip()
    for quote in ['"', "'", "«", "»"]:
        answer_clean = answer_clean.strip(quote)
    answer = answer_clean.strip().replace("\\n", "\n")

    if answer != "Информация в документах не найдена.":
        if not _is_faithful(answer, context):
            log.warning("Faithfulness check failed — suppressing answer")
            answer = "Информация в документах не найдена."

    base = _public_base_url(request)

    # Собираем ссылки на изображения СТРОГО из одного документа.
    # Приоритет: документ, чьё имя файла совпадает с темой вопроса.
    # Фоллбэк: первый чанк с картинками (самый релевантный).
    image_urls: dict[str, str] = {}
    img_marker_re = re.compile(r"\[Рисунок (\d+): ([^\]]+)\]")

    # Корни слов вопроса (первые 5 букв слов длиной ≥5)
    _q_roots = {w[:5] for w in re.findall(r"[а-яёa-z]{5,}", req.question.lower())}

    # Ищем документ с картинками, чьё имя файла лучше всего совпадает с вопросом
    _img_source_file = None
    _best_fname_score = 0
    _img_chunk_counts: dict[str, int] = {}  # файл → кол-во чанков с картинками
    for doc_text, meta in zip(docs, metas):
        fn = meta["filename"]
        if img_marker_re.search(doc_text):
            _img_chunk_counts[fn] = _img_chunk_counts.get(fn, 0) + 1
            fn_roots = {w[:5] for w in re.findall(r"[а-яёa-z]{5,}",
                        fn.replace("_", " ").replace("-", " ").lower())}
            score = len(_q_roots & fn_roots)
            if score > _best_fname_score:
                _best_fname_score = score
                _img_source_file = fn

    # Фоллбэк: документ с наибольшим числом чанков с картинками
    if not _img_source_file and _img_chunk_counts:
        _img_source_file = max(_img_chunk_counts, key=_img_chunk_counts.get)

    log.info("Image source: %s (fname_score=%d, img_chunks=%s)",
             _img_source_file, _best_fname_score, _img_chunk_counts)

    # Источники: самый релевантный документ (по совпадению имени с вопросом) первым
    _fname_relevance: dict[str, int] = {}
    for fn in _all_sources:
        fn_roots = {w[:5] for w in re.findall(r"[а-яёa-z]{5,}",
                    fn.replace("_", " ").replace("-", " ").lower())}
        _fname_relevance[fn] = len(_q_roots & fn_roots)
    sources = sorted(_all_sources, key=lambda fn: _fname_relevance.get(fn, 0), reverse=True)
    download_urls = {s: f"{base}/documents/{s}/download" for s in sources}

    # Собираем чанки и картинки из расширенного контекста (all_docs),
    # но только из файла-источника картинок
    _img_chunks: list[str] = []
    _img_metas: list[dict] = []
    if _img_source_file:
        for doc_text, meta in zip(all_docs, all_metas):
            if meta["filename"] == _img_source_file:
                _img_chunks.append(doc_text)
                _img_metas.append(meta)
                for match in img_marker_re.finditer(doc_text):
                    img_name = match.group(2)
                    marker = match.group(0)
                    if marker not in image_urls:
                        image_urls[marker] = (
                            f"{base}/documents/{meta['filename']}/images/{img_name}"
                        )

    # Привязываем иллюстрации к пунктам ответа — строго из одного документа
    if image_urls and answer != _not_found:
        answer = _attach_images_to_answer(answer, _img_chunks, _img_metas, image_urls)

    answer_html = _answer_to_html(answer, image_urls, download_urls)

    return AskResponse(
        answer=answer,
        answer_html=answer_html,
        sources=sources,
        chunks_used=len(all_docs),
        download_urls=download_urls,
        image_urls=image_urls,
        raw_chunks=all_docs[:5],
        rewritten_query=rewritten,
    )
