# LLM Engineering Masterclass

A comprehensive, hands-on masterclass on LLM Engineering - from foundations to production deployment. Conducted by [GrowAI](https://growai.in).

## Course Overview

| Detail | Value |
|--------|-------|
| Total Sessions | 20 (2 Inductions + 16 Core + 2 Spare) |
| Session Duration | 2 hours each |
| Total Teaching Hours | 36 hours |
| Schedule | Saturday + Sunday, 10:00 AM - 12:00 PM |
| Duration | 9 weekends |
| Modules | 9 |

## Session Breakdown

| Session | Day | Module | Topics |
|---------|-----|--------|--------|
| 1 | Sat, 7 Jun | Induction | GrowAI intro, mentor intro, career overview, roadmap walkthrough, setup homework |
| 2 | Fri, 13 Jun | Induction | Module walkthrough, tools overview, Q&A |
| 3 | Sat | Module 1 | AI/ML/DL/LLM hierarchy, Transformer intuition, Open-source LLM landscape, GenAI lifecycle, HuggingFace orientation, Environment setup, First model with Ollama |
| 4 | Sun | Module 2.1 | Python for AI: data structures (ChatML), functions + type hints, file I/O + JSON/YAML, error handling, Pydantic, async/await, virtual environments, project structure |
| 5 | Sat | Module 2.2 | Neural network intuition, PyTorch fundamentals, Training loop, Transformer architecture, Tokenizers & embeddings, HuggingFace Transformers library |
| 6 | Sun | Module 3 | Open-source LLM family tree, Model sizes & quantization (GGUF, AWQ), Ollama deep dive, vLLM for production, Cost economics |
| 7 | Sat | Module 4.1 | Prompt anatomy, Zero-shot/few-shot/CoT/ReAct, JSON mode & structured outputs, Pydantic + Instructor, Function calling & tool use |
| 8 | Sun | Module 4.2 + 4.3 | LangChain (LCEL, chains, memory), LlamaIndex (documents, nodes, indexes, query engines), LlamaParse, Capstone: multi-tool agent |
| 9 | Sat | Module 5 (Part 1) | RAG architecture, Embedding models, Chunking strategies, Vector databases (Milvus, ChromaDB), Index types (IVF, HNSW) |
| 10 | Sun | Module 5 (Part 2) | Hybrid search (BM25 + vector + RRF), Reranking (cross-encoders, ColBERT), Metadata filtering, Multilingual RAG, Capstone: production RAG |
| 11 | Sat | Module 6 (Part 1) | Why basic RAG fails, Multi-hop RAG, Agentic RAG (Self-RAG, CRAG), RAG evaluation |
| 12 | Sun | Module 6 (Part 2) | Knowledge graphs (Neo4j), KG construction with LLMs, Cypher query generation, GraphRAG architecture, Capstone: GraphRAG |
| 13 | Sat | Module 7 (Part 1) | Agent fundamentals (ReAct, planning, tools), LangGraph (state, nodes, edges, routing), Human-in-the-loop, Memory systems |
| 14 | Sun | Module 7 (Part 2) | Multi-agent patterns (supervisor, peer-to-peer), AutoGen, Model Context Protocol (MCP), Build custom MCP server, Capstone: multi-agent system |
| 15 | Sat | Module 8 (Part 1) | Fine-tune vs RAG vs prompting decision tree, LoRA/QLoRA, Unsloth, Training data preparation, Synthetic data generation |
| 16 | Sun | Module 8 (Part 2) | Live fine-tuning (Unsloth + Colab), LoRA config, Quantization (GGUF/AWQ), Deploy to Ollama, DPO intro |
| 17 | Sat | Module 9 (Part 1) | FastAPI for LLM APIs (async, streaming/SSE), Docker & Docker Compose, vLLM at scale, Cost optimization |
| 18 | Sun | Module 9 (Part 2) | LLM Evals (DeepEval, LLM-as-judge), Production observability, Complete architecture review, Capstone assembly |
| 19 | Sat | Spare 1 | Extended capstone / career session / overflow / deep-dive review |
| 20 | Sun | Spare 2 | Student presentations / mock interviews / course close + certificates |

## Week 1 - Foundations + Python for AI

Materials: [week-1/](./week-1/)

### Session 3 (Saturday) - How AI Actually Works + Run Your First Model

Covers the complete AI/ML/DL/LLM hierarchy with depth, Transformer intuition (tokens, attention, generation), the open-source LLM landscape (Llama, Mistral, Qwen, Phi), environment setup, and running your first model with Ollama locally. Every concept is immediately demonstrated hands-on.

**Session Outcome:** By the end of this session, you will understand the AI to ML to DL to LLM hierarchy, how Transformers work intuitively, and have run your first open-source model.

**Session Notes:** [session-3-notes.md](./week-1/session-3/session-3-notes.md)

**Homework:** [Setup Guide](./week-1/session-3/setup-guide.md) - Install Python, Ollama, and your code editor before Session 4.

### Session 4 (Sunday) - Python for AI Engineering

Python fundamentals for AI work: variables, data types, lists, dictionaries, functions, error handling. Then connecting Python to Ollama using the ChatML format, and building a simple chatbot with memory.

**Session Outcome:** By the end of this session, you can write Python functions, understand how LLMs receive conversations (ChatML), and have built a working chatbot that remembers context.

**Session Notes:** [session-4-notes.md](./week-1/session-4/session-4-notes.md)

**Homework:** Practice Python basics and get the chatbot running with a custom system prompt.

## Week 2 - Advanced Python for AI + Neural Network Foundations

Materials: [week-2/](./week-2/)

### Session 5 (Saturday) - Chatbot, Streaming, Pydantic, Async + Neural Networks

Completes Module 2.1 with all remaining Python tools for AI engineering: chatbot with memory, streaming, Pydantic validation, async/await, file I/O, and virtual environments. Then introduces neural network fundamentals (neurons, layers, training loop) and the Transformer self-attention mechanism.

**Session Outcome:** By the end of this session, you can stream AI responses, validate outputs with Pydantic, run parallel AI calls, and understand how neural networks learn.

**Teaching Notes:** [session-5-teaching-notes.md](./week-2/session-5/session-5-teaching-notes.md)

**Notebook:** [session-5-notebook.ipynb](./week-2/session-5/session-5-notebook.ipynb)

**Homework:** Build a streaming chatbot with a creative personality, validate AI data with Pydantic, optionally set up a venv locally.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed locally
- [HuggingFace](https://huggingface.co) account
- Google Colab access (for fine-tuning sessions)
- Basic programming knowledge

## Tech Stack

| Category | Tools |
|----------|-------|
| LLMs | Llama 3, Mistral, Qwen, Phi |
| Frameworks | LangChain, LlamaIndex, LangGraph |
| Vector DBs | Milvus, ChromaDB |
| Fine-tuning | Unsloth, QLoRA |
| Serving | Ollama, vLLM |
| Deployment | FastAPI, Docker, Docker Compose |
| Evals | DeepEval |
| Graph | Neo4j |

## License

MIT
