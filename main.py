"""
RAG сервис v3.5 — production pipeline
- Query rewrite (1 LLM вызов перед поиском)
- Hybrid search (vector + keyword)
- ChromaDB + LlamaIndex SentenceSplitter
- Защита от галлюцинаций
"""

import os
import re
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
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


OLLAMA_BASE_URL    = _config_value("OLLAMA_BASE_URL")
OLLAMA_LLM_MODEL   = _config_value("OLLAMA_LLM_MODEL")
OLLAMA_EMBED_MODEL = _config_value("OLLAMA_EMBED_MODEL")
LLM_API_MODE       = str(_config_value("LLM_API_MODE")).strip().lower()
STORAGE_DIR        = Path(_config_value("STORAGE_DIR"))
if not STORAGE_DIR.is_absolute():
    STORAGE_DIR = (APP_DIR / STORAGE_DIR).resolve()
FILES_DIR          = STORAGE_DIR / "files"
INDEX_FILE         = STORAGE_DIR / "index.json"
FAISS_INDEX_FILE   = STORAGE_DIR / "index.faiss"
TOP_K              = int(_config_value("TOP_K"))
CHUNK_SIZE         = int(_config_value("CHUNK_SIZE"))
CHUNK_OVERLAP      = int(_config_value("CHUNK_OVERLAP"))
BASE_URL           = _config_value("BASE_URL")
SIM_THRESHOLD      = float(_config_value("SIM_THRESHOLD"))
OLLAMA_API_KEY     = _config_value("OLLAMA_API_KEY")
OPENWEBUI_AUTH_PATH = _config_value("OPENWEBUI_AUTH_PATH")
OPENWEBUI_USER     = _config_value("OPENWEBUI_USER")
OPENWEBUI_PASSWORD = _config_value("OPENWEBUI_PASSWORD")

os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STORAGE_DIR.mkdir(exist_ok=True)
FILES_DIR.mkdir(exist_ok=True)

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".msg"}

_records: Optional[list[dict]] = None
_faiss_index = None
_indexing: dict[str, dict] = {}
_indexing_locks: dict[str, asyncio.Lock] = {}
_auth_token: Optional[str] = None
_auth_lock = asyncio.Lock()
_store_lock = asyncio.Lock()


def _load_records() -> list[dict]:
    global _records
    if _records is None:
        if INDEX_FILE.exists():
            _records = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        else:
            _records = []
    return _records


def _save_records(records: list[dict]):
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


def _load_faiss():
    import faiss

    return faiss


def _rebuild_faiss_index(records: list[dict]):
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


def _get_faiss_index():
    global _faiss_index
    records = _load_records()
    if not records:
        _faiss_index = None
        return None
    if _faiss_index is None:
        faiss = _load_faiss()
        if FAISS_INDEX_FILE.exists():
            index = faiss.read_index(str(FAISS_INDEX_FILE))
            if index.ntotal != len(records):
                log.warning(
                    "FAISS index mismatch: ntotal=%s, records=%s. Rebuilding index.",
                    index.ntotal,
                    len(records),
                )
                _rebuild_faiss_index(records)
            else:
                _faiss_index = index
        else:
            _rebuild_faiss_index(records)
    return _faiss_index


def _chunk_count() -> int:
    return len(_load_records())


def _document_chunk_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in _load_records():
        fn = record["filename"]
        counts[fn] = counts.get(fn, 0) + 1
    return counts


def _document_records(filename: str) -> list[dict]:
    return [r for r in _load_records() if r["filename"] == filename]


async def _replace_document_records(filename: str, new_records: list[dict]) -> int:
    async with _store_lock:
        current = _load_records()
        records = [r for r in current if r["filename"] != filename]
        removed = len(current) - len(records)
        records.extend(new_records)
        _rebuild_faiss_index(records)
        _save_records(records)
        return removed


async def _delete_document_records(filename: str) -> int:
    async with _store_lock:
        current = _load_records()
        records = [r for r in current if r["filename"] != filename]
        removed = len(current) - len(records)
        if removed == 0:
            return 0
        _rebuild_faiss_index(records)
        _save_records(records)
        return removed


def _search_records(query_embedding: list[float], top_k: int) -> dict:
    index = _get_faiss_index()
    records = _load_records()
    if index is None or not records:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    vec = np.asarray([query_embedding], dtype="float32")
    scores, indices = index.search(vec, min(top_k, len(records)))

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    distances: list[float] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        record = records[int(idx)]
        ids.append(record["id"])
        docs.append(record["text"])
        metas.append({"filename": record["filename"]})
        distances.append(float(1 - score))

    return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [distances]}


def _load_documents(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".msg":
        try:
            import extract_msg
            from llama_index.core.schema import Document
        except ImportError as e:
            raise RuntimeError(
                "MSG support requires the 'extract-msg' package"
            ) from e

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
            r = await c.post(
                signin_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()

        token = (
            data.get("token")
            or data.get("access_token")
            or data.get("data", {}).get("token")
        )
        if not token and "token=" in r.headers.get("set-cookie", ""):
            cookie = r.headers["set-cookie"].split("token=", 1)[1]
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


async def _embed(text: str):
    async with httpx.AsyncClient(timeout=120) as c:
        headers = await _api_headers()
        r = await c.post(
            _embed_url(),
            headers=headers,
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
        )
        if r.status_code == 401 and OPENWEBUI_PASSWORD and OPENWEBUI_USER:
            headers = await _api_headers(force_refresh=True)
            r = await c.post(
                _embed_url(),
                headers=headers,
                json={"model": OLLAMA_EMBED_MODEL, "input": text},
            )
        if r.status_code == 400:
            return None
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return data["data"][0]["embedding"]
        emb = data["embeddings"]
        return emb[0] if isinstance(emb[0], list) else emb


async def _chat(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=600) as c:
        headers = await _api_headers()
        mode = _api_mode()
        if mode in {"openai", "openwebui"}:
            r = await c.post(
                _chat_url(),
                headers=headers,
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.0,
                    "top_p": 0.85,
                    "max_tokens": 400,
                },
            )
            if r.status_code == 401 and OPENWEBUI_PASSWORD and OPENWEBUI_USER:
                headers = await _api_headers(force_refresh=True)
                r = await c.post(
                    _chat_url(),
                    headers=headers,
                    json={
                        "model": OLLAMA_LLM_MODEL,
                        "messages": messages,
                        "stream": False,
                        "temperature": 0.0,
                        "top_p": 0.85,
                        "max_tokens": 400,
                    },
                )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        else:
            r = await c.post(
                _chat_url(),
                headers=headers,
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.85,
                        "top_k": 30,
                        "repeat_penalty": 1.15,
                        "num_ctx": 8192
                    }
                },
            )
            if r.status_code == 401 and OPENWEBUI_PASSWORD and OPENWEBUI_USER:
                headers = await _api_headers(force_refresh=True)
                r = await c.post(
                    _chat_url(),
                    headers=headers,
                    json={
                        "model": OLLAMA_LLM_MODEL,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": 0.0,
                            "top_p": 0.85,
                            "top_k": 30,
                            "repeat_penalty": 1.15,
                            "num_ctx": 8192
                        }
                    },
                )
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
        if not content:
            raise ValueError("LLM вернул пустой ответ")
        return content


async def _rewrite_query(q: str) -> str:
    return q  # отключено для qwen3


async def _run_indexing(filename: str, dest: Path):
    lock = _indexing_locks.setdefault(filename, asyncio.Lock())
    async with lock:
        _indexing[filename] = {
            "status": "indexing", "chunks_done": 0, "chunks_total": 0,
            "error": None, "started_at": datetime.utcnow().isoformat(), "finished_at": None,
        }
        try:
            from llama_index.core.node_parser import SentenceSplitter

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None,
                lambda: _load_documents(dest)
            )
            if not docs:
                raise ValueError("No text extracted from file")

            splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            nodes = await loop.run_in_executor(
                None,
                lambda: splitter.get_nodes_from_documents(docs)
            )

            total = len(nodes)
            _indexing[filename]["chunks_total"] = total
            log.info(f"[{filename}] {total} chunks (SentenceSplitter)")

            done = 0
            new_records: list[dict] = []
            for i, node in enumerate(nodes):
                text = node.text.strip()
                if not text:
                    continue
                emb = await _embed(text)
                if emb is None:
                    log.warning(f"[{filename}] skipping chunk {i} (embed 400)")
                    continue
                new_records.append({
                    "id": f"{filename}_{i}",
                    "filename": filename,
                    "text": text,
                    "embedding": _normalize_embedding(emb),
                })
                done += 1
                _indexing[filename]["chunks_done"] = done
                if done % 20 == 0:
                    log.info(f"[{filename}] {done}/{total}")

            await _replace_document_records(filename, new_records)

            _indexing[filename].update({
                "status": "ready", "chunks_done": done,
                "finished_at": datetime.utcnow().isoformat(),
            })
            log.info(f"[{filename}] done: {done} chunks")

        except Exception as e:
            log.error(f"[{filename}] failed: {e}")
            _indexing[filename].update({
                "status": "error", "error": str(e),
                "finished_at": datetime.utcnow().isoformat(),
            })


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"FAISS index: {_chunk_count()} chunks")
    yield

app = FastAPI(title="RAG Ollama API v3.5", version="3.5.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    download_urls: dict[str, str]
    raw_chunks: list[str]

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
    total = _chunk_count()
    docs = len(_document_chunk_counts())
    return StatusResponse(
        total_chunks=total, total_documents=docs,
        llm_model=OLLAMA_LLM_MODEL, embed_model=OLLAMA_EMBED_MODEL,
        ollama_url=OLLAMA_BASE_URL,
    )

@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    counts = _document_chunk_counts()
    for fn in _indexing:
        counts.setdefault(fn, 0)
    result = []
    for fn, n in sorted(counts.items()):
        st = _indexing.get(fn, {}).get("status", "ready" if n > 0 else "unknown")
        result.append(DocumentInfo(
            filename=fn, chunks=n, indexing_status=st,
            download_url=f"{BASE_URL}/documents/{fn}/download",
        ))
    return result

@app.post("/documents", response_model=UploadResponse, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Unsupported type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED))}")
    if _indexing.get(file.filename, {}).get("status") == "indexing":
        raise HTTPException(409, f"'{file.filename}' is already being indexed")
    dest = FILES_DIR / file.filename
    dest.write_bytes(await file.read())
    background_tasks.add_task(_run_indexing, file.filename, dest)
    return UploadResponse(filename=file.filename, status="indexing_started")

@app.get("/documents/{filename}/download")
async def download_document(filename: str):
    path = FILES_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"File '{filename}' not found")
    return FileResponse(path, filename=filename)

@app.get("/documents/{filename}/status", response_model=IndexingStatus)
async def document_status(filename: str):
    info = _indexing.get(filename)
    if info is None:
        chunks = len(_document_records(filename))
        if chunks:
            return IndexingStatus(
                filename=filename, status="ready",
                chunks_done=chunks, chunks_total=chunks,
                progress_pct=100.0, error=None,
                started_at=None, finished_at=None,
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

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    if _indexing.get(filename, {}).get("status") == "indexing":
        raise HTTPException(409, f"'{filename}' is currently being indexed")
    removed = await _delete_document_records(filename)
    if not removed:
        raise HTTPException(404, f"Document '{filename}' not found in index")
    (FILES_DIR / filename).unlink(missing_ok=True)
    _indexing.pop(filename, None)
    return {"filename": filename, "chunks_removed": removed}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if _chunk_count() == 0:
        raise HTTPException(503, "Index is empty. Upload documents first via POST /documents")

    top_k = req.top_k or TOP_K

    # 1. Query rewrite
    rewritten = await _rewrite_query(req.question)
    rewritten = rewritten.replace('"', '')
    # Убираем нелатинские/некириллические символы (китайский и т.д.)
    rewritten = re.sub(r'[^-ɏЀ-ӿ\s]', '', rewritten).strip()
    # Для поиска используем оригинальный вопрос — rewrite может добавлять мусор
    search_query = req.question

    # 2. Векторный поиск
    try:
        q_emb = await _embed(search_query)
        if q_emb is None:
            raise ValueError("embed returned None")
    except Exception as e:
        raise HTTPException(502, f"Embedding API error: {e}")

    results = _search_records(_normalize_embedding(q_emb), top_k)

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return AskResponse(
            answer="Информация в документах не найдена.",
            sources=[], chunks_used=0, download_urls={}, raw_chunks=[]
        )

    best_sim = 1 - distances[0]
    log.info(f"Best similarity: {best_sim:.3f}")
    if best_sim < SIM_THRESHOLD:
        return AskResponse(
            answer="Информация в документах не найдена.",
            sources=[], chunks_used=0, download_urls={}, raw_chunks=[]
        )

    # 3. Используем top-k векторных чанков, но отсекаем одиночные "хвосты" из чужих документов
    doc_counts: dict[str, int] = {}
    for meta in metas:
        fn = meta["filename"]
        doc_counts[fn] = doc_counts.get(fn, 0) + 1

    primary_filename = metas[0]["filename"]
    all_docs = []
    all_metas = []
    for doc, meta in zip(docs, metas):
        fn = meta["filename"]
        if fn == primary_filename or doc_counts.get(fn, 0) >= 2:
            all_docs.append(doc)
            all_metas.append(meta)

    context = "\n\n".join(
        f"[CHUNK {i} | {m['filename']}]\n{d}"
        for i, (d, m) in enumerate(zip(all_docs, all_metas))
    )
    sources = list(dict.fromkeys(m["filename"] for m in all_metas))

    # 4. LLM ответ
    try:
        answer = await _chat([
            {
                "role": "system",
                "content": (
                    "Отвечай только на русском языке.\n"
                    "Контекст ниже — единственный источник правды.\n"
                    "Строго запрещено придумывать шаги, роли, ограничения, числа, лимиты, сроки и требования, которых нет в контексте.\n"
                    "Строго запрещено объединять информацию из разных фрагментов, если связь между ними не очевидна из текста.\n"
                    "Если в контексте нет прямого ответа на вопрос, ответь ровно одной фразой: Информация в документах не найдена.\n"
                    "Если ответ есть, дай 3-5 коротких пунктов по шагам, только по контексту.\n"
                    "Отвечай по существу, без введения и без рассуждений.\n"
                    "Не пиши фразы вроде: 'Давайте разберем', 'Вот пошаговая инструкция', 'Based on the provided documentation', 'Okay'.\n"
                    "Не переводи термины на английский и не смешивай языки.\n"
                    "Если есть сомнение, лучше ответь: Информация в документах не найдена."
                )
            },
            {
                "role": "user",
                "content": f"Контекст:\n\n{context}\n\nВопрос: {req.question}"
            },
        ])
    except Exception as e:
        raise HTTPException(502, f"Chat API error: {e}")

    log.info(f"Raw answer: {answer[:200]!r}")

    answer_clean = answer.strip()
    for q in ['"', "'", "«", "»"]:
        answer_clean = answer_clean.strip(q)
    answer = answer_clean.strip().replace("\\n", "\n")

    if answer != "Информация в документах не найдена.":
        def normalize(t: str) -> str:
            return re.sub(r"\s+", " ", t.lower()).strip()

        ctx_norm = normalize(context)
        ans_norm = normalize(answer)

        # Мягкая проверка: ответ должен опираться на лексику из контекста.
        if ans_norm:
            words = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{3,}", ans_norm) if len(w) > 3]
            overlap = sum(1 for w in words if w in ctx_norm)
            if words and overlap < max(2, int(len(words) * 0.3)):
                log.warning(f"Hallucination: low overlap ({overlap}/{len(words)})")
                answer = "Информация в документах не найдена."
    download_urls = {s: f"{BASE_URL}/documents/{s}/download" for s in sources}
    return AskResponse(
        answer=answer, sources=sources,
        chunks_used=len(all_docs),
        download_urls=download_urls,
        raw_chunks=all_docs[:3]
    )
