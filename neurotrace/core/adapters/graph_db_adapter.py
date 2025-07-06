"""
Graph Database Adapter Module.

This module provides adapter functions for converting neurotrace Message objects
into graph database compatible formats. It handles the transformation of messages
into graph nodes and relationship edges, suitable for graph database storage
and querying.
"""

from typing import Any, Dict, List
from neurotrace.core.schema import Message, MessageMetadata


def to_graph_node(msg: "Message") -> Dict[str, Any]:
    """
    Convert a Message object to a graph database node format.

    This function transforms a neurotrace Message into a dictionary format
    suitable for creating nodes in a graph database. It includes the message's
    core properties and flattens metadata into the node properties.

    Args:
        msg (Message): The Message object to convert into a graph node.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - id: The unique identifier for the node
            - labels: List of node labels (["Message"])
            - properties: Dict containing:
                - role: The message role (user/ai/system)
                - content: The message text content
                - timestamp: ISO formatted timestamp
                - Additional properties from message metadata

    Example:
        >>> msg = Message(id="123", content="Hello", role="user")
        >>> node = to_graph_node(msg)
        >>> print(node["labels"])  # ["Message"]
        >>> print(node["properties"]["role"])  # "user"
    """
    return {
        "id": msg.id,
        "labels": ["Message"],
        "properties": {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            **msg.metadata.model_dump()
        }
    }


def graph_edges_from_related_ids(msg: "Message") -> List[Dict[str, str]]:
    """
    Generate graph relationship edges from a message's related IDs.

    This function creates relationship edges between the given message and its
    related messages, as specified in the message's metadata. Each relationship
    is of type "RELATED_TO".

    Args:
        msg (Message): The Message object to generate relationship edges for.

    Returns:
        List[Dict[str, str]]: A list of edge dictionaries, each containing:
            - from: The source message ID (current message)
            - to: The target message ID (related message)
            - type: The relationship type ("RELATED_TO")

    Example:
        >>> msg = Message(id="123", metadata=MessageMetadata(related_ids=["456"]))
        >>> edges = graph_edges_from_related_ids(msg)
        >>> print(edges[0])  # {"from": "123", "to": "456", "type": "RELATED_TO"}
    """
    return [{"from": msg.id, "to": rid, "type": "RELATED_TO"} for rid in msg.metadata.related_ids or []]
