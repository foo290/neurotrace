"""
Test module for the graph database adapter functionality.

This module verifies the conversion of Message objects to graph database nodes and edges,
ensuring proper serialization of message content, metadata, and relationship mappings
between messages in the graph structure.
"""

from neurotrace.core.schema import Message, MessageMetadata
from neurotrace.core.adapters.graph_db_adapter import to_graph_node, graph_edges_from_related_ids


def test_to_graph_node_properties():
    msg = Message(
        role="ai",
        content="Graph ready!",
        metadata=MessageMetadata(
            source="chat",
            tags=["memory"]
        )
    )

    node = to_graph_node(msg)

    assert node["id"] == msg.id
    assert node["labels"] == ["Message"]
    props = node["properties"]
    assert props["role"] == "ai"
    assert props["content"] == "Graph ready!"
    assert props["tags"] == ["memory"]


def test_graph_edges_from_related_ids():
    msg = Message(
        role="user",
        content="Follow-up message",
        metadata=MessageMetadata(related_ids=["msg123", "msg456"])
    )

    edges = graph_edges_from_related_ids(msg)

    assert len(edges) == 2
    assert edges[0]["from"] == msg.id
    assert edges[0]["to"] == "msg123"
    assert edges[0]["type"] == "RELATED_TO"
