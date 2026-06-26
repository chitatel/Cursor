# RAG Ollama API v5.0 — справочник эндпоинтов

Документация для подключения внешних систем (1С, Telegram-бот, портал, веб-приложения).

- **Базовый URL примера:** `http://app174:8000` (замените на свой)
- **Аутентификация:** нет (предполагается доверенная локальная сеть)
- **Формат запросов:** JSON (`application/json`) или `multipart/form-data` для загрузки файлов
- **Формат ответов:** JSON (`application/json`), UTF-8
- **OpenAPI/Swagger:** доступен по адресу `GET /docs` (интерактивная документация)
- **OpenAPI JSON:** `GET /openapi.json`

---

## Содержание

1. [Опрос системы](#1-опрос-системы)
   - `GET /status` — состояние сервиса
   - `GET /categories` — список категорий
   - `GET /documents` — список документов
   - `GET /logs` — лог запросов
2. [Главное: задать вопрос](#2-главное-задать-вопрос)
   - `POST /ask` — RAG-ответ по корпусу
3. [Управление документами](#3-управление-документами)
   - `POST /documents` — загрузить файл
   - `POST /documents/from-url` — импорт статьи с портала по URL
   - `POST /documents/from-portal-bulk` — массовый импорт из портала
   - `POST /documents/prepare` — подготовить файл к индексации (без загрузки)
   - `POST /documents/{filename}/reindex` — переиндексировать
   - `POST /documents/reindex-all` — переиндексировать всё
   - `PUT /documents/{filename}/category` — сменить категорию
   - `DELETE /documents/{filename}` — удалить документ
   - `DELETE /documents?category=X` — удалить категорию целиком
4. [Файлы и статусы](#4-файлы-и-статусы)
   - `GET /documents/{filename}/download` — скачать оригинал
   - `GET /documents/{filename}/images/{image_name}` — скачать картинку
   - `GET /documents/{filename}/status` — статус индексации
   - `GET /files/...` — статические файлы (документы и картинки)
   - `GET /prepared/...` — подготовленные docx
5. [Модели данных](#5-модели-данных)
6. [Коды ошибок](#6-коды-ошибок)

---

## 1. Опрос системы

### `GET /status`

Состояние сервиса: количество документов, чанков, имена моделей.

**Ответ 200:**
```json
{
  "total_chunks": 1842,
  "total_documents": 94,
  "llm_model": "gemma3:latest",
  "embed_model": "nomic-embed-text:v1.5",
  "ollama_url": "http://localhost:11434"
}
```

Используется как health-check.

---

### `GET /categories`

Список всех существующих категорий в индексе.

**Ответ 200:**
```json
["Документооборот", "Бухгалтерия", "Portal"]
```

---

### `GET /documents`

Список всех документов.

**Параметры (query):**
- `category` — фильтр по категории (опционально)

**Ответ 200:**
```json
[
  {
    "filename": "kb_0076_Отпуск как сделать правильный выбор.md",
    "chunks": 18,
    "indexing_status": "ready",
    "download_url": "http://app174:8000/files/kb_0076_Отпуск как сделать правильный выбор.md",
    "category": "Portal"
  }
]
```

`indexing_status`: `ready` | `indexing` | `failed` | `unknown`.

---

### `GET /logs`

Последние записи лога `/ask`, новые сверху. Логируются только запросы к `/ask`.

**Параметры (query):**
- `limit` — 1..5000 (по умолчанию 200)
- `date_from` — ISO-строка: `2026-06-25` или `2026-06-25T12:00:00Z`
- `date_to` — то же самое

**Ответ 200:**
```json
[
  {
    "timestamp": "2026-06-25T13:35:13Z",
    "question": "оформление отпуска",
    "category": null,
    "top_k": null,
    "best_similarity": 0.823,
    "answer": "1. График отпусков утверждается...",
    "sources": ["kb_0076_Отпуск...md"],
    "chunks_used": 18,
    "duration_ms": 7184
  }
]
```

---

## 2. Главное: задать вопрос

### `POST /ask`

Получить RAG-ответ по проиндексированному корпусу.

**Запрос:**
```json
{
  "question": "как оформить отпуск",
  "top_k": 12,
  "category": "Portal"
}
```

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `question` | string | да | Вопрос на русском |
| `top_k` | int | нет | Сколько чанков взять. По умолчанию из config (12) |
| `category` | string | нет | Искать только в этой категории |

**Ответ 200:**
```json
{
  "answer": "1. График отпусков утверждается...\n2. Основной отпуск...",
  "answer_html": "<p>Вам может подойти:</p><ul>...</ul><ol><li>...</li></ol>",
  "sources": ["kb_0076_Отпуск...md", "kb_0105_Заявка...md"],
  "chunks_used": 12,
  "download_urls": {
    "kb_0076_Отпуск...md": "http://app174:8000/files/kb_0076_..."
  },
  "image_urls": {
    "[Рисунок 1: img_001.png]": "http://app174:8000/files/kb_0076_..._images/img_001.png"
  },
  "raw_chunks": ["...", "..."],
  "rewritten_query": null
}
```

**Поля:**
- `answer` — текст ответа (нумерованный список или фраза «Информация в документах не найдена.»).
- `answer_html` — готовая HTML-разметка для встраивания в форму. Содержит блок «Вам может подойти:» со ссылками на исходники + сам ответ + картинки.
- `sources` — список исходных файлов, отсортированный по релевантности (первый — основной).
- `chunks_used` — сколько чанков ушло в LLM.
- `download_urls` — ссылки на скачивание исходных файлов.
- `image_urls` — карта `маркер → URL картинки`. Маркеры вида `[Рисунок N: img_001.png]` встречаются в `answer`.
- `raw_chunks` — сырые чанки, использованные для ответа (для отладки).
- `rewritten_query` — если включён `QUERY_REWRITE_ENABLED`, тут переписанный вопрос; иначе `null`.

**Пример (curl):**
```bash
curl -X POST http://app174:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"как оформить отпуск"}'
```

**Пример (1С):**
```bsl
Соединение = Новый HTTPСоединение("app174", 8000);
Запрос = Новый HTTPЗапрос("/ask");
Запрос.Заголовки.Вставить("Content-Type", "application/json");
Запрос.УстановитьТелоИзСтроки("{""question"":""как оформить отпуск""}");
Ответ = Соединение.ОтправитьДляОбработки(Запрос);
Данные = ПрочитатьJSON(Новый ЧтениеJSON(Ответ.ПолучитьТелоКакСтроку()));
```

**Особенности:**
- Если поиск ничего не нашёл (низкая similarity, нет лексических совпадений), возвращается ответ «Информация в документах не найдена.» с пустыми `sources`.
- Если вопрос «широкий» (короткий, без слов «как/где/когда/...»), ответ объединяет шаги из всех релевантных документов. Если «узкий» — только из одного раздела.
- Картинки привязываются к пунктам ответа автоматически по совпадению слов.

---

## 3. Управление документами

### `POST /documents`

Загрузить файл и поставить на индексацию. Возвращается сразу (HTTP 202), индексация идёт в фоне — следить через `GET /documents/{filename}/status`.

**Запрос:** `multipart/form-data`

| Поле | Тип | Обязательно |
|------|-----|-------------|
| `file` | binary | да |
| `category` | string | нет |

Поддерживаемые расширения: `.pdf`, `.docx`, `.txt`, `.md`, `.msg`.

**Ответ 202:**
```json
{
  "filename": "Положение об отпусках.docx",
  "status": "indexing_started",
  "category": "Документооборот"
}
```

**Пример (curl):**
```bash
curl -X POST http://app174:8000/documents \
  -F "file=@instruction.docx" \
  -F "category=Документооборот"
```

---

### `POST /documents/from-url`

Скачать статью с корпоративной Базы знаний (`start.gk-osnova.ru/kb/single?pid=NNN`) и проиндексировать. Авторизация — HTTP Basic из `config.json` (`KB_PORTAL_USER` / `KB_PORTAL_PASSWORD`).

**Запрос:**
```json
{
  "url": "https://start.gk-osnova.ru/kb/single?pid=76",
  "category": "Portal"
}
```

**Ответ 202:**
```json
{
  "filename": "kb_0076_Отпуск как сделать правильный выбор.md",
  "status": "indexing_started",
  "category": "Portal"
}
```

Если статья с тем же `pid` уже была — старая версия (md + папка картинок) удаляется и заменяется новой.

---

### `POST /documents/from-portal-bulk`

Массовый импорт всех статей с портала по диапазону pid.

**Запрос:**
```json
{
  "pid_min": 1,
  "pid_max": 500,
  "category": "Portal"
}
```

**Ответ 202:**
```json
{
  "status": "bulk_import_started",
  "pid_min": 1,
  "pid_max": 500,
  "category": "Portal"
}
```

Прогресс — в лог сервиса (`[KB bulk] pid=N saved → ...`). На каждый pid пауза 0.3 сек, чтобы не нагружать портал.

---

### `POST /documents/prepare`

Подготовить файл (нормализовать текст для RAG) без загрузки в индекс. Возвращает ссылки на скачивание подготовленного docx + текстовой версии + отчёта о предупреждениях. Полезно для предварительного просмотра/правки перед заливкой.

**Запрос:** `multipart/form-data` с полем `file`.

**Ответ 200:**
```json
{
  "source_filename": "instruction.pdf",
  "prepared_filename": "instruction_prepared.docx",
  "status": "ok",
  "warnings": [],
  "download_url": "http://app174:8000/prepared/instruction_prepared.docx",
  "text_url": "http://app174:8000/prepared/instruction_prepared.md",
  "report_url": "http://app174:8000/prepared/instruction_prepared_report.json"
}
```

---

### `POST /documents/{filename}/reindex`

Переиндексировать существующий документ без повторной загрузки. Удаляет старые чанки и картинки, запускает индексацию заново.

**Параметры (query):**
- `category` — если задана, обновится. Если не задана — сохраняется текущая. Пустая строка сбрасывает.

**Ответ 200:**
```json
{
  "filename": "instruction.docx",
  "status": "reindexing_started",
  "category": "Документооборот"
}
```

---

### `POST /documents/reindex-all`

Переиндексировать **все** документы. Категории сохраняются.

**Ответ 200:**
```json
{
  "started": ["doc1.docx", "doc2.pdf"],
  "skipped_already_indexing": []
}
```

---

### `PUT /documents/{filename}/category`

Сменить категорию документа без переиндексации (быстро).

**Запрос:**
```json
{"category": "Бухгалтерия"}
```

Пустая строка или `null` сбрасывает категорию.

**Ответ 200:**
```json
{
  "filename": "instruction.docx",
  "category": "Бухгалтерия",
  "chunks_updated": 18
}
```

---

### `DELETE /documents/{filename}`

Удалить один документ: файл с диска, папку картинок, все чанки из индекса.

**Ответ 200:**
```json
{"filename": "instruction.docx", "chunks_removed": 18}
```

---

### `DELETE /documents?category=X`

Удалить **все** документы указанной категории.

**Параметры (query):**
- `category` — обязательно

**Ответ 200:**
```json
{
  "category": "Portal",
  "documents_removed": ["kb_0001_...md", "kb_0002_...md"],
  "chunks_removed": 432
}
```

---

## 4. Файлы и статусы

### `GET /documents/{filename}/download`

Скачать исходный файл документа.

### `GET /documents/{filename}/images/{image_name}`

Скачать картинку, извлечённую из документа.

Альтернативная (более частая) ссылка: `GET /files/{stem}_images/{image_name}` — отдаётся через статический mount. Именно эти URL подставляются в `image_urls` ответа `/ask`.

### `GET /documents/{filename}/status`

Статус индексации.

**Ответ 200:**
```json
{
  "filename": "instruction.docx",
  "status": "indexing",
  "chunks_done": 8,
  "chunks_total": 18,
  "progress_pct": 44.4,
  "error": null,
  "started_at": "2026-06-25T13:00:00Z",
  "finished_at": null
}
```

`status`: `indexing` | `ready` | `failed`.

### `GET /files/...` и `GET /prepared/...`

Статические маунты — отдают содержимое каталогов `storage/files/` и `storage/prepared/` напрямую. Через них работают все ссылки в `download_urls` и `image_urls`.

Пример: `GET /files/kb_0076_Отпуск..._images/img_001.png` → файл из `storage/files/kb_0076_Отпуск..._images/img_001.png`.

---

## 5. Модели данных

### AskRequest
```typescript
{ question: string, top_k?: int, category?: string }
```

### AskResponse
```typescript
{
  answer: string,
  answer_html: string,
  sources: string[],
  chunks_used: int,
  download_urls: { [filename]: url },
  image_urls: { [marker]: url },
  raw_chunks: string[],
  rewritten_query: string | null
}
```

### UploadResponse
```typescript
{ filename: string, status: string, category: string | null }
```

### DocumentInfo
```typescript
{
  filename: string,
  chunks: int,
  indexing_status: "ready" | "indexing" | "failed" | "unknown",
  download_url: string,
  category: string | null
}
```

### IndexingStatus
```typescript
{
  filename: string,
  status: "indexing" | "ready" | "failed",
  chunks_done: int,
  chunks_total: int,
  progress_pct: float,
  error: string | null,
  started_at: string | null,    // ISO 8601
  finished_at: string | null
}
```

### LoadFromUrlRequest
```typescript
{ url: string, category?: string }
```

### BulkImportRequest
```typescript
{ pid_min: int, pid_max: int, category?: string }
```

### CategoryUpdateRequest
```typescript
{ category?: string }
```

### StatusResponse
```typescript
{
  total_chunks: int,
  total_documents: int,
  llm_model: string,
  embed_model: string,
  ollama_url: string
}
```

### PrepareResponse
```typescript
{
  source_filename: string,
  prepared_filename: string,
  status: "ok" | "warnings",
  warnings: string[],
  download_url: string,
  text_url: string,
  report_url: string
}
```

---

## 6. Коды ошибок

| Код | Когда | Тело |
|-----|-------|------|
| `200` | OK | результат |
| `202` | Принято в фоновую обработку (upload, reindex, bulk) | `{filename, status, ...}` |
| `400` | Некорректный запрос (неподдерживаемое расширение, плохой URL, плохие параметры) | `{"detail": "..."}` |
| `404` | Документ/файл/картинка не найдены | `{"detail": "..."}` |
| `409` | Конфликт: документ уже индексируется или категория содержит индексируемые документы | `{"detail": "..."}` |
| `500` | Внутренняя ошибка (ошибка извлечения текста, неподготовленный документ) | `{"detail": "..."}` |
| `502` | Ошибка внешнего сервиса (LLM API, портал, эмбеддер недоступен) | `{"detail": "..."}` |

Формат тела ошибки — стандартный FastAPI:
```json
{"detail": "Описание проблемы"}
```

---

## Шаблон интеграции (псевдокод)

```python
import requests

BASE = "http://app174:8000"

# 1. Загрузить документ
with open("instruction.docx", "rb") as f:
    r = requests.post(f"{BASE}/documents",
        files={"file": f}, data={"category": "Документооборот"})
fname = r.json()["filename"]

# 2. Подождать готовности
while True:
    s = requests.get(f"{BASE}/documents/{fname}/status").json()
    if s["status"] == "ready": break
    if s["status"] == "failed": raise RuntimeError(s["error"])
    time.sleep(1)

# 3. Спросить
r = requests.post(f"{BASE}/ask",
    json={"question": "как оформить отпуск"})
ans = r.json()
print(ans["answer"])
for src in ans["sources"]:
    print(" —", ans["download_urls"][src])
```

---

## Куда смотреть подробнее

- **Интерактивный Swagger:** `http://app174:8000/docs` — там можно тыкать запросы прямо из браузера.
- **OpenAPI JSON для генерации клиента:** `http://app174:8000/openapi.json`.
- **Лог сервиса** — пишется в stdout процесса (по умолчанию). Уровень INFO. Для каждого запроса `/ask` запись в `storage/ask_log.jsonl`.
