from app.answering import LEGAL_SYSTEM_PROMPT, generate_grounded_answer


def test_prompt_forbids_invented_legal_rules() -> None:
    assert "Не выдумывай нормы" in LEGAL_SYSTEM_PROMPT
    assert "Не дополняй найденные нормы знаниями из памяти модели" in LEGAL_SYSTEM_PROMPT
    assert "будь взрослым" in LEGAL_SYSTEM_PROMPT
    assert "Спокойная сильная позиция" in LEGAL_SYSTEM_PROMPT


def test_prompt_requires_insufficient_source_disclosure() -> None:
    required_phrase = "В предоставленных источниках недостаточно данных для уверенного вывода"
    assert required_phrase in LEGAL_SYSTEM_PROMPT
    assert required_phrase in generate_grounded_answer("Есть ли нарушение?", [])
