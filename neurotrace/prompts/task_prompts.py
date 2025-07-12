from langchain.prompts import PromptTemplate

PROMPT_VECTOR_AND_GRAPH_SUMMARY = PromptTemplate.from_template(
    """
You are a multi-format summarization assistant.

Given the input message below, generate two types of outputs:

1. VECTOR_SUMMARY:
- A semantically rich, concise, and meaningful representation.
- Optimized for use in vector similarity search (RAG, embeddings, etc.)
- Avoids repetition and preserves high-level intent/context.

2. GRAPH_SUMMARY:
- A structured version with explicit factual statements.
- Optimized for triplet (subject–predicate–object) extraction for use in knowledge graphs.
- Use standalone, unambiguous, and declarative sentences.

MESSAGE:
{message}

Return in the following format:

VECTOR_SUMMARY:
<your concise semantic summary here>

GRAPH_SUMMARY:
<your clear factual summary here>

RETURN A JSON WITH KEYS:

vector_summary,
graph_summary
"""
)


PROMPT_GENERAL_SUMMARY = PromptTemplate.from_template(
    """
You are a summarization assistant.
Given the input message below, generate a concise and meaningful summary.

MESSAGE:
{message}

Return a string that is semantically rich, concise, and meaningful.
"""
)


PROMPT_GRAPH_SUMMARY = PromptTemplate.from_template(
    """
You are a summarization assistant who can generate Graph summaries.

Given the input message below, generate outputs:

GRAPH_SUMMARY:
- A structured version with explicit factual statements.
- Optimized for triplet (subject–predicate–object) extraction for use in knowledge graphs.
- Use standalone, unambiguous, and declarative sentences.

MESSAGE:
{message}

Make sure to generate high quality summaries that are clear and concise.
This summary will be later used to extract triplets to form a graph in following format:
subject - Relation - Object

So make sure the summary is structured in a way that it can be easily parsed into triplets.

Return in the following format:

GRAPH_SUMMARY:
<your clear factual summary here>


Return a string.
"""
)


PROMPT_TRIPLETS_EXTRACTOR = PromptTemplate.from_template(
    """
Extract all factual triplets from the following input. Each triplet should follow the form:

Subject - Relation - Object

Be precise and unambiguous. If the object is implied or missing, use an empty string.

Input:
{text}

Output:
Subject - Relation - Object

Return a JSON List of Lists in the format:
[["Subject1", "Relation1", "Object1"], ["Subject2", "Relation2", "Object2"], ...]

Return an empty list if no triplets can be extracted.
"""
)


PROMPT_SUMMARISE_VECTOR_AND_GRAPH_MEMORY = PromptTemplate(
    input_variables=["vector_memory", "graph_memory"],
    template="""
You are an intelligent assistant summarizing relevant context retrieved from two memory systems:

1. Vector Memory – Unstructured, semantically retrieved passages from long-term memory.
2. Graph Memory – Structured relationships and factual connections between concepts.

Your task is to read the memory contents below and produce a concise, meaningful summary that captures the key ideas, facts, or relationships.

---

Vector Memory:
{vector_memory}

---

Graph Memory:
{graph_memory}

---

Instructions:
- If both memory sources contain relevant content, synthesize them into a unified summary.
- If only one of them has relevant content, summarize based on that source alone.
- If neither contains useful information, return this exact sentence:

"No relevant context found in memory."
""",
)
