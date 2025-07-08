from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from neurotrace.core.schema import Message
from neurotrace.core.vector_memory import VectorMemoryAdapter


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def mock_embedding_model():
    return MagicMock()


@pytest.fixture
def adapter(mock_vector_store, mock_embedding_model):
    return VectorMemoryAdapter(vector_store=mock_vector_store, embedding_model=mock_embedding_model)


@pytest.fixture
def sample_messages():
    return [
        Message(role="human", content="What is the capital of France?", metadata={"id": "msg1"}),
        Message(role="ai", content="The capital of France is Paris.", metadata={"id": "msg2"}),
    ]


def test_add_messages_calls_add_documents(adapter, mock_vector_store, sample_messages):
    adapter.add_messages(sample_messages)
    assert mock_vector_store.add_documents.called
    documents = mock_vector_store.add_documents.call_args[0][0]
    assert len(documents) == len(sample_messages)
    assert all(isinstance(doc, Document) for doc in documents)


def test_search_returns_valid_messages(adapter, mock_vector_store):
    mock_doc = Document(page_content="The capital of France is Paris.", metadata={"role": "ai", "id": "msg2"})
    mock_vector_store.similarity_search.return_value = [mock_doc]

    results = adapter.search("capital of France", k=1)

    mock_vector_store.similarity_search.assert_called_once_with(query="capital of France", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Message)
    assert results[0].content == "The capital of France is Paris."


def test_delete_calls_underlying_delete(adapter, mock_vector_store):
    mock_vector_store.delete = MagicMock()
    adapter.delete(["msg1", "msg2"])
    mock_vector_store.delete.assert_called_once_with(["msg1", "msg2"])


def test_delete_raises_if_not_supported(mock_vector_store, mock_embedding_model):
    if hasattr(mock_vector_store, "delete"):
        delattr(mock_vector_store, "delete")

    adapter = VectorMemoryAdapter(vector_store=mock_vector_store, embedding_model=mock_embedding_model)

    with pytest.raises(NotImplementedError):
        adapter.delete(["msg1"])
