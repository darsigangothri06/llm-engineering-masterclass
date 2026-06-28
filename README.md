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

### Session 6 (Sunday) - PyTorch & Transformers

PyTorch fundamentals (tensors, nn.Module), writing a training loop from scratch and watching the model learn y=2x, and the full Transformer architecture (multi-head attention, positional encoding, decoder-only).

**Session Outcome:** By the end of this session, you can write a training loop in PyTorch and understand the full Transformer architecture behind every LLM.

**Teaching Notes:** [session-6-teaching-notes.md](./week-2/session-6/session-6-teaching-notes.md)

**Notebook:** [session-6-notebook.ipynb](./week-2/session-6/session-6-notebook.ipynb)

**Homework:** Run the training loop notebook. Change the learning rate to 0.1 and to 0.001. Compare.

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
