from app.chunking import chunk_legal_text


def test_structured_text_preserves_article_numbers_and_section() -> None:
    text = """\
Глава 1. Общие положения
Статья 1. Основные правила
1. Первое положение применяется к отношениям сторон.
2. Второе положение устанавливает обязанность уведомления.

Статья 12.1. Специальное правило
1) Условие специального правила.
"""

    chunks = chunk_legal_text(text, max_chars=180, overlap=20)

    article_chunks = [chunk for chunk in chunks if chunk.article_number]
    assert {chunk.article_number for chunk in article_chunks} == {"1", "12.1"}
    assert all(chunk.section_title == "Глава 1. Общие положения" for chunk in article_chunks)
    assert "Статья 12.1" in next(
        chunk.text for chunk in article_chunks if chunk.article_number == "12.1"
    )


def test_unstructured_text_uses_overlapping_fallback_chunks() -> None:
    text = " ".join(f"слово{i}" for i in range(90))

    chunks = chunk_legal_text(text, max_chars=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunk.article_number is None for chunk in chunks)
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert chunks[0].text.split()[-1] in chunks[1].text
