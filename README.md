# Юридический GPT-помощник по законодательству РФ

Рабочий MVP RAG-системы для предварительного юридического анализа. Пользователь
описывает ситуацию, приложение извлекает релевантные фрагменты из собственной базы
документов и передает модели только найденный контекст. Ответ содержит ссылки на
фрагменты, степень уверенности и предупреждения.

Это не модель, которая должна помнить все законодательство. Юридически значимый
ответ нельзя строить на памяти LLM: редакции актов меняются, а непроверенная ссылка
на статью создает риск ошибочного действия. В этом проекте источником вывода служат
загруженные документы и их retrieved-чанки.

## Возможности

- `POST /ingest/file`: загрузка UTF-8 `.txt`/`.md` по локальному пути либо текста
  в JSON, структурный legal chunking, OpenAI embeddings и сохранение в pgvector.
- `POST /ask`: семантический поиск по cosine distance и grounded-ответ по
  retrieved-фрагментам с источниками.
- `GET /health`: состояние API и соединения с PostgreSQL.
- CLI для инициализации БД, загрузки файла и вопроса.
- HNSW cosine-индекс pgvector для поиска.
- Безопасный stub для будущего подключения официального источника.

## Стек

Python 3.11+, FastAPI, PostgreSQL + pgvector, SQLAlchemy 2, OpenAI API,
Pydantic, python-dotenv, Docker Compose и pytest.

## Быстрый запуск

Из каталога `legal-rag-rf`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` укажите действительный API-ключ:

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

`OPENAI_EMBEDDING_DIMENSIONS` задает размерность столбца pgvector при создании
таблиц. После создания БД не изменяйте ее без миграции и повторного построения
embeddings. Модель ответа настраивается через переменную окружения.

Поднимите PostgreSQL с расширением pgvector и создайте схему:

```bash
docker compose up -d db
python scripts/init_db.py
```

Запустите API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Проверка доступности:

```bash
curl http://localhost:8000/health
```

## Загрузка первого документа

Для локального файла поместите разрешенно полученный UTF-8 документ, например,
в `data/raw/contract.md`, затем выполните:

```bash
python scripts/ingest_file.py data/raw/contract.md \
  --title "Абонентский договор фитнес-клуба" \
  --source-url "local-upload" \
  --doc-type "contract" \
  --jurisdiction "RF" \
  --version-date "2026-05-27"
```

Текст также можно загрузить через API без создания файла:

```bash
curl -X POST http://localhost:8000/ingest/file \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Статья 1. Текст официально полученного документа...",
    "title": "Тестовый источник",
    "source_url": "local-upload",
    "doc_type": "legal_document",
    "jurisdiction": "RF",
    "version_date": "2026-05-27"
  }'
```

В продуктивной базе вместо тестового текста загружайте полный документ из
проверяемого источника с корректной датой редакции.

## Задать вопрос

Через API:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Какие мои права могли быть нарушены?", "top_k": 8}'
```

Через CLI:

```bash
python scripts/ask_cli.py "Какие мои права могли быть нарушены?"
```

Ответ API имеет форму:

```json
{
  "answer": "А. Краткий вывод ... [S1]",
  "sources": [
    {
      "title": "Название документа",
      "source_url": "local-upload",
      "doc_type": "legal_document",
      "version_date": "2026-05-27",
      "chunk_id": "uuid",
      "excerpt": "Фрагмент..."
    }
  ],
  "confidence": "medium",
  "warnings": []
}
```

## Структура

```text
app/
  main.py                 FastAPI endpoints
  models.py               documents/chunks и pgvector index
  chunking.py             legal-aware нарезка с fallback overlap
  embeddings.py           OpenAI embeddings
  retrieval.py            cosine search
  answering.py            строгий grounded legal prompt
  ingest.py               ingestion transaction
  source_adapters/        адаптеры источников
scripts/                  init_db, ingest_file, ask_cli
tests/                    unit-тесты
```

## Источники и официальный портал

`app/source_adapters/pravo_publication.py` намеренно является stub. Подключение
официальных публикаций РФ следует реализовать только через разрешенный API,
официальный экспорт или другой явно допустимый способ получения данных с фиксацией
реквизитов документа и редакции. Проект не выполняет агрессивный scraping и не
предполагает обход правил сайта.

Для добавления источника реализуйте `BaseSourceAdapter.fetch()`, `normalize()` и
`get_metadata()`, затем передавайте нормализованный текст в ingestion-поток.

## Ограничения и безопасность

- Система не является адвокатом и выдает предварительный анализ.
- Наличие похожего чанка не доказывает применимость нормы к фактам.
- Перед юридически значимыми действиями нужно сверять актуальную редакцию акта.
- При отсутствии достаточных фрагментов ответ обязан сообщить о недостатке данных.
- В срочной опасной ситуации следует обращаться к адвокату, полиции или
  экстренным службам, а не полагаться на RAG-ответ.

Полные правила поведения описаны в [LEGAL_ASSISTANT_POLICY.md](LEGAL_ASSISTANT_POLICY.md).

## Тесты и проверки

```bash
pytest
python -c "from app.main import app; print(app.title)"
docker compose config
```

Тесты не вызывают OpenAI API и не требуют запущенного PostgreSQL: retrieval
проверяется детерминированным mock embedding provider. Для end-to-end проверки
нужны PostgreSQL/pgvector и ключ OpenAI.

## Справочные ссылки OpenAI

- [Embeddings API reference](https://platform.openai.com/docs/api-reference/embeddings)
- [Responses API reference](https://platform.openai.com/docs/api-reference/responses)
- [text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small)
