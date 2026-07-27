import streamlit as st
import torch
import os
import uuid
import time
import io
import concurrent.futures
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

from langchain_text_splitters import RecursiveCharacterTextSplitter
import gc
from transformers import (
    AutoModelForImageTextToText, 
    AutoProcessor, 
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread

from helper import get_vision_token_cost

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding

IMAGE_DIR = "./extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Database Setup
@st.cache_resource
def init_database():
    client = QdrantClient(path="./local_rag_db")
    
    if not client.collection_exists(collection_name="document_collection"):
        client.create_collection(
            collection_name="document_collection",
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE), 
                "colbert": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0) 
                )
            }
        )
    return client

# Model Loading
@st.cache_resource(show_spinner="Initializing Models...")
def load_models():
    model_id = "Qwen/Qwen3.5-2B"
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    dense_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    colbert_embedder = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
    
    return model, processor, dense_embedder, colbert_embedder

# --- 3. Hackathon Fast Parsing & Extraction ---
@st.cache_data(show_spinner=False)
def hackathon_parse_pdf(file_bytes, filename):
    """
    Blazing fast hybrid parser. Uses PyMuPDF for text/image extraction.
    Falls back to multi-threaded Tesseract OCR ONLY for scanned files.
    Cached so live demos are completely instant.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)
    
    def process_single_page(page_num):
        page = doc[page_num]
        text = page.get_text().strip()
        
        if len(text) < 50:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
            text = pytesseract.image_to_string(img)
            
        image_paths = []
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            img_filename = f"{uuid.uuid4().hex}.{image_ext}"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(img_path)
            
        return page_num, text, image_paths

    parsed_pages = {}
    
    # Multi-threaded OCR to speed up bad PDFs
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_page = {executor.submit(process_single_page, i): i for i in range(page_count)}
        for future in concurrent.futures.as_completed(future_to_page):
            p_num, p_text, p_images = future.result()
            parsed_pages[p_num] = {"text": p_text, "images": p_images}

    return [parsed_pages[i] for i in range(page_count)]


def process_and_index_pdf(uploaded_file, client, processor, dense_embedder, colbert_embedder):
    file_bytes = uploaded_file.read()
    
    start_time = time.time()
    parsed_pages = hackathon_parse_pdf(file_bytes, uploaded_file.name)
    parse_time = time.time() - start_time
    st.toast(f"Parsed {len(parsed_pages)} pages in {parse_time:.2f} seconds!")

    # Micro-Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents, metadatas, ids = [], [], []

    for page_num, page_data in enumerate(parsed_pages):
        parent_text = page_data["text"]
        if not parent_text:
            continue

        page_images = page_data["images"]
        # Take the first image found on the page to pass to the Vision model
        primary_image = page_images[0] if page_images else ""

        # Base metadata 
        base_meta = {
            "source": uploaded_file.name,
            "Header 1": f"Page {page_num + 1}", # Page Numbers
            "Header 2": "",
            "image_path": primary_image,
            "parent_text": parent_text 
        }
        
        sub_chunks = text_splitter.split_text(parent_text)
        
        for i, sub_chunk in enumerate(sub_chunks):
            chunk_meta = base_meta.copy()
            
            # only attaches the image to the first sub-chunk 
            if i > 0 and "image_path" in chunk_meta:
                chunk_meta["image_path"] = ""
            
            chunk_meta["token_count"] = len(processor.tokenizer.encode(sub_chunk, add_special_tokens=False))
            
            documents.append(sub_chunk)
            metadatas.append(chunk_meta) 
            ids.append(uuid.uuid4().hex)

    if documents:
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            dense_vectors = list(dense_embedder.embed(batch_docs))
            colbert_vectors = list(colbert_embedder.embed(batch_docs))
            
            points = [
                models.PointStruct(
                    id=batch_ids[idx],
                    vector={
                        "dense": dense_vectors[idx],
                        "colbert": colbert_vectors[idx]
                    },
                    payload={"text": batch_docs[idx], **batch_metas[idx]}
                )
                for idx in range(len(batch_docs))
            ]
            
            client.upload_points(
                collection_name="document_collection",
                points=points
            )

# Query Reformulation ; if it starts tweaking just replace this with the old one in test.py
def reformulate_query(current_prompt, messages, model, processor):
    # gets recent history
    user_questions = [msg['content'] for msg in messages[:-1] if msg['role'] == 'user']
    history = "\n".join([f"Past Question: {q}" for q in user_questions[-3:]]) if user_questions else "No past history."
    
    sys_prompt = (
        "You are a lightning-fast medical search query optimizer. Your job is to transform the user's Latest Query into a dense, keyword-rich search string for a database. "
        "1. Resolve any pronouns (it, this, he) using the Past Questions. "
        "2. Add 2 to 3 highly relevant synonyms, alternative phrasing, or technical medical terms to the query to ensure a broad search match. "
        "Return ONLY the optimized, expanded query string. Do not include quotes, explanations, or conversational text."
    )
    user_prompt = f"{history}\n\nLatest Query: {current_prompt}\n\nOptimized Search Query:"
    
    format_msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    
    formatted_prompt = processor.apply_chat_template(format_msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.2)
    
    input_length = inputs['input_ids'].shape[1]
    rewritten_query = processor.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    # original prompt if the model hallucinates an empty string
    return rewritten_query if rewritten_query else current_prompt

# UI stuff
st.set_page_config(page_title="Local RAG", page_icon="⚡")
st.title("Multimodal Document AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

client = init_database()
model, processor, dense_embedder, colbert_embedder = load_models()

with st.sidebar:
    st.header("Knowledge Base")

    if client.collection_exists("document_collection"):
        records, _ = client.scroll(collection_name="document_collection", limit=10000, with_vectors=False)
        unique_sources = set([record.payload.get("source") for record in records if record.payload and "source" in record.payload])
    else:
        unique_sources = set()

    if unique_sources:
        st.write("**Currently Indexed:**")
        for source in unique_sources:
            st.caption(f"- {source}")
    else:
        st.write("Database is empty.")

    st.divider()
    
    uploaded_files = st.file_uploader("Upload New PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process & Index"):
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            for uploaded_file in uploaded_files:
                process_and_index_pdf(uploaded_file, client, processor, dense_embedder, colbert_embedder)
                st.success(f"{uploaded_file.name} added to database!")
            st.rerun()

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "images" in msg and msg["images"]:
                cols = st.columns(len(msg["images"]))
                for col, img_path in zip(cols, msg["images"]):
                    if os.path.exists(img_path):
                        col.image(img_path, use_container_width=True)

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Understanding context..."):
        search_query = reformulate_query(prompt, st.session_state.messages, model, processor)

    with st.spinner("Searching, reranking, and packing context..."):
        query_dense = list(dense_embedder.embed([search_query]))[0]
        query_colbert = list(colbert_embedder.embed([search_query]))[0]

        qdrant_results = client.query_points(
            collection_name="document_collection",
            prefetch=models.Prefetch(
                query=query_dense,
                using="dense",
                limit=50 
            ),
            query=query_colbert,
            using="colbert",
            limit=10, 
            with_payload=True
        )
        
        points = qdrant_results.points if hasattr(qdrant_results, 'points') else qdrant_results

        MAX_CONTEXT_TOKENS = 1500  
        HARD_VRAM_LIMIT = 2500     

        top_results = []
        current_token_count = 0
        seen_parents = set()
        
        for res in points:
            meta = res.payload
            score = res.score
            
            full_text = meta.get("parent_text", meta.get("text", ""))
            
            if full_text in seen_parents:
                continue
            seen_parents.add(full_text)
            
            text_tokens_approx = len(full_text) // 4
            
            img_path = meta.get("image_path")
            if img_path and os.path.exists(img_path):
                image_budget = get_vision_token_cost(img_path)
            else:
                image_budget = 0

            chunk_tokens = text_tokens_approx + image_budget
            
            if current_token_count + chunk_tokens > MAX_CONTEXT_TOKENS:
                remaining_budget = MAX_CONTEXT_TOKENS - current_token_count
                if remaining_budget > 50 and not meta.get("image_path"):
                    truncated_doc = full_text[:remaining_budget * 4] + "\n... [Content truncated to fit VRAM limit]"
                    top_results.append((score, truncated_doc, meta))
                break 
                
            top_results.append((score, full_text, meta))
            current_token_count += chunk_tokens

        retrieved_scores = [res[0] for res in top_results]
        retrieved_docs = [res[1] for res in top_results]
        retrieved_metas = [res[2] for res in top_results]

        if not retrieved_docs:
            st.warning("No highly relevant information found in the indexed documents.")
            st.stop()

    retrieved_images = []
    formatted_chunks = []
    
    with st.expander("📄 View Retrieved Sources", expanded=False):
        for score, doc, meta in zip(retrieved_scores, retrieved_docs, retrieved_metas):
            source = meta.get("source", "Unknown Source")
            h1 = meta.get("Header 1", "")
            section = h1 if h1 else "General"
            
            citation_tag = f"[{source} | {section}]"
            
            img_path = meta.get("image_path")
            has_image = img_path and os.path.exists(img_path)
            
            if has_image:
                safe_img = Image.open(img_path).convert("RGB")
                safe_img.thumbnail((768, 768))
                retrieved_images.append(safe_img)
                img_tag = f"\n[Refer to Image {len(retrieved_images)}]"
            else:
                img_tag = ""

            chunk_block = (
                f"--- BEGIN CHUNK ---\n"
                f"Citation Tag: {citation_tag}{img_tag}\n"
                f"Content:\n{doc}\n"
                f"--- END CHUNK ---"
            )
            formatted_chunks.append(chunk_block)
            
            st.markdown(f"**{citation_tag}** *(Fusion Score: `{score:.4f}`)*\n\n{doc}")
            if has_image:
                st.image(img_path, caption=f"Extracted artifact from {source}", use_container_width=True)
            st.divider()
        
    context = "\n\n".join(formatted_chunks)
    
    system_prompt = (
        "You are an assistant that answers questions using only the provided text and images. "
        "Start your answer directly. "
        "Always append the relevant [Citation Tags] at the end of every answer generated."
        "If the information is missing, say 'I do not know'."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    recent_history = st.session_state.messages[-3:-1] 
    for msg in recent_history:
        content = msg["content"]
        if msg["role"] == "assistant" and len(content) > 1200:
            content = content[:1200] + "\n...[Previous answer truncated to save VRAM]..."
        messages.append({"role": msg["role"], "content": content})
    
    user_prompt_text = f"Analyze the following context chunks and images to answer the question.\n\n{context}\n\nQuestion:\n{prompt}\n\nCRITICAL: You MUST cite your sources. Append the exact [Citation Tags] to the end of your response."

    user_content = []
    for img in retrieved_images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": user_prompt_text})

    messages.append({"role": "user", "content": user_content})

    formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=formatted_prompt,
        images=retrieved_images if retrieved_images else None,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, temperature=0.3)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
        
    del inputs
    if 'outputs' in locals():
        del outputs
    gc.collect()
    torch.cuda.empty_cache()
        
    valid_image_paths = [
        meta.get("image_path") 
        for meta in retrieved_metas 
        if meta.get("image_path") and os.path.exists(meta.get("image_path"))
    ]

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "images": valid_image_paths  
    })