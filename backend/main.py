"""
backend/main.py
FastAPI Gateway with Server-Sent Events (SSE) Streaming & Cache Optimization
"""
import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
import docx
from dotenv import load_dotenv

from embeddings import embedding_engine
from vector_store import vector_store
from cache import cache_manager
from rag_engine import rag_engine

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure Qdrant collection exists
    print("🚀 Starting FastAPI Server & Initializing DBs...")
    await vector_store.init_collection()
    yield
    print("🛑 Shutting down FastAPI Server...")


app = FastAPI(title="Low-Latency RAG Engine", lifespan=lifespan)

# Allow Cross-Origin Resource Sharing (CORS) for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    prompt: str


@app.get("/health")
async def health_check():
    return {"status": "online", "message": "API Gateway operating normally"}


@app.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Ingests .txt, .pdf, or .docx files, extracts text content,
    generates embeddings locally, and upserts to Qdrant.
    """
    try:
        file_bytes = await file.read()
        filename = file.filename.lower()
        extracted_text = ""

        # 1. Parse PDF files
        if filename.endswith(".pdf"):
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            extracted_text = "\n\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )

        # 2. Parse Word (.docx) files
        elif filename.endswith(".docx"):
            doc_file = io.BytesIO(file_bytes)
            doc = docx.Document(doc_file)
            extracted_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        # 3. Parse Plain Text (.txt) files
        elif filename.endswith(".txt"):
            extracted_text = file_bytes.decode("utf-8")

        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload .txt, .pdf, or .docx"
            )

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file contains no extractable text.")

        # Chunking Logic (Splits by paragraphs / 500 chars)
        chunks = [c.strip() for c in extracted_text.split("\n\n") if c.strip()]
        if not chunks:
            chunks = [extracted_text[i:i+500] for i in range(0, len(extracted_text), 500)]

        # Generate vectors locally
        vectors = embedding_engine.generate_batch_vectors(chunks)

        # Upsert into Qdrant
        await vector_store.upsert_chunks(
            session_id=session_id,
            chunks=chunks,
            vectors=vectors,
            doc_name=file.filename
        )

        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}'",
            "chunks_stored": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 1. New endpoint to fetch chat history when the UI loads
@app.get("/history/{session_id}")
async def get_history_endpoint(session_id: str):
    history = await cache_manager.get_history(session_id)
    return {"session_id": session_id, "history": history}


# 2. Updated /chat endpoint to save messages automatically
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    prompt = request.prompt

    # Record User Message in History
    await cache_manager.append_history(session_id, "User", prompt)

    # Check Redis Cache
    cached_response = await cache_manager.get(session_id, prompt)
    if cached_response:
        await cache_manager.append_history(session_id, "Assistant", cached_response)
        async def stream_cached():
            yield cached_response
        return StreamingResponse(stream_cached(), media_type="text/plain")

    # Streaming Generator
    async def token_stream_generator():
        full_response = []
        async for token in rag_engine.generate_rag_stream(session_id, prompt):
            full_response.append(token)
            yield token
        
        complete_text = "".join(full_response)
        if complete_text:
            # Save complete response to Cache & History
            await cache_manager.set(session_id, prompt, complete_text)
            await cache_manager.append_history(session_id, "Assistant", complete_text)

    return StreamingResponse(token_stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)