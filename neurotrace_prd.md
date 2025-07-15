# Neurotrace — Product Requirements Document (PRD)

**Title:** Neurotrace: Hybrid Memory Library for LangChain Agents\
**Owner:** Nitin Sharma\
**Status:** Draft\
**Last Updated:** July 15, 2025

---

## 📌 1. Overview

**Neurotrace** is a memory library designed for integration with **LangChain agents**. It provides a dual-layer memory model comprising **short-term buffer memory** and a **long-term hybrid RAG system**. The purpose of Neurotrace is to give conversational agents persistent, intelligent memory that improves over time and enables contextual understanding and recall.

---

## 🌟 2. Goals

- Build a robust memory architecture for agents.
- Enable **real-time recall** and **intelligent storage** of information during conversations.
- Support **custom metadata-rich message structures** for future filtering, compression, or semantic tracing.
- Use **hybrid retrieval** (vector + graph RAG) for deeper and more accurate contextual reasoning.

---

## 👤 3. Target Users

- Developers building AI agents with LangChain.
- Researchers exploring memory augmentation in LLMs.
- Enterprises looking to deploy context-aware AI assistants.

---

## 🔍 4. Features

### ✅ Core Components

1. **Custom Message Format**

   - Supports additional metadata (timestamps, tags, source, intent, etc.).
   - Acts as the canonical data structure across memory layers.

2. **Short-Term Memory (STM)**

   - Buffer memory to hold recent messages for fast recall.
   - Token-aware memory windowing to manage overflow.

3. **Long-Term Memory (LTM)**

   - Composed of:
     - **Vector RAG**: Embedding-based retrieval of semantically relevant messages.
     - **Graph RAG**: Knowledge graph-based reasoning using extracted triplets and relationships.

4. **Hybrid Retrieval Flow**

   - Query first checks STM.
   - If insufficient, triggers vector + graph retrieval.
   - Retrieved content is fused into a final context window.

5. **Real-Time Memory Ingestion**

   - During chat:
     - Messages are embedded and stored in the vector DB.
     - Triplets are extracted and pushed to a graph DB.
   - Graph relationships evolve organically as conversations happen.

6. **LangChain Integration**

   - All memory modules derive from or wrap around `LangChain` memory classes.
   - Plug-and-play compatibility with LangChain agents.

---

## 🧪 5. Success Metrics

| Metric                                       | Target                   |
| -------------------------------------------- | ------------------------ |
| Integration time with LangChain agent        | < 15 mins                |
| Average recall latency (STM + LTM)           | < 300ms                  |
| Precision of context retrieval (manual eval) | > 90%                    |
| Triplet extraction success rate              | > 95% per batch          |
| STM memory overflow handling                 | Always under token limit |

---

## 🧰 6. Technical Details

### 🔧 Dependencies:

- `LangChain`
- `Neo4j` or other graph DB (for Graph RAG)
- `FAISS`, `Chroma`, or `Pinecone` (for Vector RAG)
- `OpenAI`, `Anthropic`, or local LLM endpoint
- Optional: Triplet extractor model (`openie`, `spaCy`, `LLM-based`)

### 💾 Storage Layer:

- Vector DB: Stores embedded messages
- Graph DB: Stores dynamic triplet-based relationships
- Optional: PostgreSQL or SQLite for message logs

---

## ⚠️ 7. Assumptions & Constraints

- LLM must be able to parse structured graph data if passed as text or JSON.
- Triplet extraction must be accurate enough to prevent graph pollution.
- Token limits must be enforced at all stages (STM window, final context fusion).
- Developers must provide their own LLM API keys and DB config.

---

## 🖼️ 8. Mock UX (Example Flow)

```
User: "I’m working on a project that uses Kafka to stream simulation events."

Agent stores:
✔ Message → Vector DB (semantically indexed)
✔ Triplet → (Project)-[:USES]->(Kafka) → Graph DB

Later...
User: "Remind me what tools I use in that project?"

→ STM check → miss
→ Vector + Graph RAG → retrieve Kafka + simulation context
→ Fused context → LLM answers
```

---

## 🗓️ 9. Timeline

| Milestone                    | Date          |
| ---------------------------- | ------------- |
| Initial STM + LTM separation | ✅ June 2025   |
| Hybrid retrieval working     | ✅ July 2025   |
| Fusion model integration     | ✅ July 2025   |
| PRD Finalization             | 🟡 July 2025  |
| Open-source candidate        | ⬜ August 2025 |

---

## 🔍 10. Open Questions

- Should we add a relevance scoring mechanism for triplets?
- Do we need a UI or visualization layer for the graph memory?
- Should memory support versioning or rollback?
