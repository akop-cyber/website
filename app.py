from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import tempfile, os, uuid

from loader import Loader
from chunker import Chunker
from embedder import Embedder
from vector import VectorStorage
from retriever import Retriever

app = FastAPI()

# ── CORS: allow your website's domain to call this API ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://akop-cyber.github.io"],          # 🔒 Replace "*" with your website URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")

# In-memory session store  {session_id: {"store": ..., "embedder": ...}}
# For production, use Redis or a database instead
sessions: dict = {}


# ── 1. Upload & Index PDF ────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Receives a PDF, builds the vector index, returns a session_id."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save upload to a temp file (Loader expects a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text    = Loader(tmp_path).load()
        chunks  = Chunker().chunker(text)
        embedder = Embedder()
        vectors  = embedder.embed(chunks)
        store    = VectorStorage(dimension=len(vectors[0]))
        store.add(vectors, chunks)
    finally:
        os.unlink(tmp_path)          # clean up temp file

    # Store session server-side, return only the key to the client
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"store": store, "embedder": embedder}

    return {"session_id": session_id, "message": "PDF indexed. Ready to chat!"}


# ── 2. Chat ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message:    str
    history:    list  # [{"role": "user"|"assistant", "content": "..."}]

from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(req: ChatRequest):
    """Retrieves context, streams the LLM response back."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    store    = session["store"]
    embedder = session["embedder"]

    retriever     = Retriever(store, embedder, k=3)
    context_chunks = retriever.retrieve(req.message)

    if not context_chunks:
        return {"response": "I couldn't find relevant information in the document."}

    context_text  = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a research assistant. Answer questions using only the provided context. "
        "If the answer isn't there, say you don't know. Do not hallucinate."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {req.message}"})

    # Stream tokens back as Server-Sent Events (SSE)
    def token_stream():
        for token in client.chat_completion(messages, max_tokens=512, stream=True):
            text = token.choices[0].delta.content
            if text:
                # SSE format: "data: <chunk>\n\n"
                yield f"data: {text}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
