# Multimodal Local RAG (Sub-3GB VRAM)

> **Private, offline, and lightweight Multimodal Document AI running entirely on consumer GPUs.** 

An end-to-end local Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **Qwen3.5-2B**, and vector databases (**Qdrant** / **ChromaDB**). This application extracts both text and visual artifacts (charts, diagrams, tables) from your PDF documents, indexes them with hybrid search, and feeds them directly into a 4-bit vision-language model—**all while consuming less than 3 GB of VRAM.**

---

## Key Features

* ** Extreme Efficiency (<3GB VRAM Target):** Uses 4-bit NF4 quantization (`bitsandbytes`), `bfloat16` precision, dynamic context window management, and strict token budget calculations to run on budget/entry-level GPUs.
* ** Truly Multimodal Retrieval & Generation:** Extracts visual elements (tables, diagrams, figures) directly from PDFs. The vision-language model (`Qwen3.5-2B`) analyzes both context chunks and image artifacts side-by-side.
* ** Advanced Hybrid Retrieval (ColBERT + Dense):** Integrates **Qdrant** with dual-vector indexing (`bge-small-en-v1.5` dense embeddings + `colbertv2.0` late-interaction multi-vectors)
* ** Blazing-Fast PDF Ingestion:** Uses a multi-threaded hybrid PDF parser (PyMuPDF/Fitz + Tesseract OCR fallback) or 
* ** Context-Aware Query Reformulation:** Automatically resolves pronouns ("it", "this", "he") and expands search queries using conversation history to maximize retrieval recall.
* ** Streaming Chat Interface with Strict Citations:** Includes full chat history, expandable source citations, visual image artifacts in chat history, and streamed model outputs using `TextIteratorStreamer`.

---

##  Architecture & VRAM Engineering

```text
  [PDF Upload] ──> [PyMuPDF Parser] ──> Extract Text & Images
                                                        │
                                                        ▼
[Query Input] ──> [Context Reformulation] ──> [Hybrid Vector DB Search]
                                            (Dense + ColBERT / Reranker)
                                                        │
                                                        ▼
[Streamed Answer] <── [Qwen3.5-2B (4-bit)] <── [Context & Image Packing]
```

### How it stays under 3 GB VRAM:
1. **4-Bit Model Quantization:** `Qwen/Qwen3.5-2B` is loaded in 4-bit NormalFloat (`nf4`) quantization using `bitsandbytes`.
2. **Vision Token Cost Calculation:** Uses a custom `get_vision_token_cost` helper that calculates image patch grids ($28 \times 28$ pixel blocks) before passing images into the prompt, ensuring hard context budget limits (`MAX_CONTEXT_TOKENS = 1500`) are respected.
3. **Dynamic Context Truncation:** Text chunks and previous chat histories are dynamically truncated if token allocation approaches hardware limits (`HARD_VRAM_LIMIT = 2500`).
4. **Active Cache Clearing:** Explicit garbage collection (`gc.collect()`) and CUDA cache flushing (`torch.cuda.empty_cache()`) are executed after each generation turn.

---

##  Prerequisites

* **OS:** Windows, Linux, or macOS (Apple Silicon)
* **Python:** 3.10+
* **CUDA Support:** Recommended (NVIDIA GPU with $\ge$ 4 GB total VRAM recommended, though runtime usage remains <3 GB).
* **Tesseract OCR (Optional):** Required only if running PyMuPDF fast OCR parsing on scanned PDF files.

---

##  Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/multimodal-local-rag.git](https://github.com/your-username/multimodal-local-rag.git)
cd multimodal-local-rag
```

### 2. Install PyTorch with CUDA
Before installing dependencies, install PyTorch compiled for your specific CUDA version (e.g., CUDA 12.6):
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

---

##  Project Structure

```text
├── app.py       
├── helper.py            # Vision token estimation math (Qwen-VL patch grid math)
├── requirements.txt     # Python dependencies
├── local_rag_db/        # Persistent local vector database (ChromaDB / Qdrant)
└── extracted_images/    # Directory where extracted PDF images/figures are saved
```

---

##  Dependencies (`requirements.txt`)

```text
transformers>=4.39.0
bitsandbytes>=0.46.1
accelerate>=0.28.0
sentence-transformers>=2.5.0
faiss-cpu>=1.8.0
chromadb
docling
streamlit
accelerate
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# install this too btw
```

---

##  Configuration & Model Customization

* **LLM Engine:** Default is set to `Qwen/Qwen3.5-2B` (`AutoModelForImageTextToText`).
* **Dense Embeddings:** Default is set to `BAAI/bge-small-en-v1.5` or `mixedbread-ai/mxbai-embed-large-v1`.
* **Late-Interaction Embeddings:** Default is set to `colbert-ir/colbertv2.0`.
* **Vector DB Path:** Databases are persisted locally under `./local_rag_db`.

---

##  License

Distributed under the MIT License. See `LICENSE` for more information.
