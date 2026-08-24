from infra.knowledge.sparse_encoder import LocalSparseEncoder, _terms


def test_sparse_encoder_is_stable() -> None:
    encoder = LocalSparseEncoder()

    first = encoder.encode("话费查询 HF")
    second = encoder.encode("话费查询 HF")

    assert first.indices == second.indices
    assert first.values == second.values
    assert first.indices
    assert all(value > 0 for value in first.values)


def test_sparse_terms_include_chinese_bigrams() -> None:
    terms = _terms("话费查询")

    assert "话费" in terms
    assert "费查" in terms
    assert "查询" in terms
