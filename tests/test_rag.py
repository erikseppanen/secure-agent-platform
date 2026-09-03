import pytest

import app.rag as rag


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(12))

    chunks = rag.chunk_text(text, chunk_size_words=5, overlap_words=2)

    assert chunks == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",
    ]


def test_vector_literal() -> None:
    assert rag._vector_literal([0.1, 0.2, -0.3]) == "[0.1,0.2,-0.3]"


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        rag.chunk_text("one two three", chunk_size_words=3, overlap_words=3)
