from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple, Union

from langchain.llms.base import BaseLLM
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_store import GraphStore
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAI

from neurotrace.core.utils import safe_json_loads, strip_json_code_block
from neurotrace.prompts.task_prompts import PROMPT_TRIPLETS_EXTRACTOR


class GraphTripletIndexerBase(ABC): ...


class GraphTripletIndexer(GraphTripletIndexerBase):
    def __init__(self, llm: Union[BaseLLM, BaseChatModel]):
        self.llm = llm
        self.prompt = PROMPT_TRIPLETS_EXTRACTOR

    def extract(self, graph_summary: str) -> List[Tuple[str, str]]:
        prompt_text = self.prompt.format(text=graph_summary)
        response = self.llm.invoke(prompt_text).content.strip()
        response = strip_json_code_block(response).lower()

        return safe_json_loads(response, return_type=list)  # noqa


class BaseGraphMemoryAdapter(ABC): ...


# class CustomGraphCypherQAChain(GraphCypherQAChain):


class GraphMemoryAdapter(BaseGraphMemoryAdapter):
    def __init__(
        self,
        llm: Union[BaseLLM, BaseChatModel],
        graph_database: GraphStore,
        triplets_indexer: GraphTripletIndexerBase = None,
    ):
        self.llm = llm
        self.graph = graph_database

        self.triplets_indexer = triplets_indexer or GraphTripletIndexer(self.llm)
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm, graph=self.graph, verbose=True, allow_dangerous_requests=True  # Prints the generated Cypher
        )

    def insert_triplets(
        self, triples, sender: Literal["user", "agent"] = "user", timestamp: str = None, tags: List[str] = None
    ):
        if not triples:
            print("No triplets to insert")
            return

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if len(triples) > 3:
            print("Too many triplets", triples)
            return

        s, r, o = triples
        query = f"""
        MERGE (a:Entity {{name: $s}})
          ON CREATE SET a.created_at = $timestamp

        MERGE (b:Entity {{name: $o}})
          ON CREATE SET b.created_at = $timestamp

        MERGE (a)-[r:`{r.upper().replace(" ", "_")}`]->(b)
          ON CREATE SET r.created_at = $timestamp, r.source = $sender
          ON CREATE SET r.tags = $tags
        """
        self.graph.query(
            query, params={"s": s, "o": o, "r": r, "timestamp": timestamp, "sender": sender, "tags": tags or []}
        )

    def add_conversation(
        self, summarised_text: str, sender: Literal["user", "agent"] = "agent", tags: List[str] = None
    ):
        """
        Add a conversation to the graph memory.

        Args:
            sender (Literal["user", "agent"]): The sender of the message.
            :param summarised_text:
        """
        triplets = self.triplets_indexer.extract(summarised_text)

        for triplet in triplets:
            self.insert_triplets(triplet, sender=sender, tags=tags)

    def ask_graph(self, query: str) -> Dict[str, Any]:
        """
        Ask a question to the graph memory and get an answer.

        Args:
            query (str): The question to ask.

        Returns:
            str: The answer from the graph memory.
        """
        return self.qa_chain.invoke({"query": query})

    def get_all_relation_types(self) -> list[str]:
        result = self.graph.query("CALL db.relationshipTypes()")
        # Output is like: [{'relationshipType': 'WORKS_AT'}, ...]
        return [record["relationshipType"] for record in result]
