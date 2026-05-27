from app.retrieval import VectorCandidate, rank_candidates


class MockEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


def test_rank_candidates_uses_query_embedding_and_similarity_order() -> None:
    provider = MockEmbeddingProvider()
    candidates = [
        VectorCandidate("unrelated", "не связано", [0.0, 1.0]),
        VectorCandidate("closest", "точный фрагмент", [1.0, 0.0]),
        VectorCandidate("partial", "частичный фрагмент", [0.7, 0.3]),
    ]

    ranked = rank_candidates("вопрос пользователя", candidates, provider, top_k=2)

    assert provider.queries == ["вопрос пользователя"]
    assert [candidate.identifier for candidate, _score in ranked] == ["closest", "partial"]
    assert ranked[0][1] == 1.0
