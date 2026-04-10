from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import tempfile, os, uuid, logging

from loader import Loader
from chunker import Chunker
from embedder import Embedder
from vector import VectorStorage
from retriever import Retriever

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MODELS = [
    ("Qwen/Qwen2.5-72B-Instruct"),
    ("Qwen/Qwen2.5-72B-Instruct"),
    ("meta-llama/Llama-3.2-3B-Instruct"),
    ("meta-llama/Llama-3.2-3B-Instruct"),
    ("meta-llama/Llama-3.2-3B-Instruct"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("mistralai/Mistral-7B-Instruct-v0.3"),
    ("mistralai/Mistral-7B-Instruct-v0.3"),
    ("HuggingFaceH4/zephyr-7b-beta"),
]




sessions: dict = {}



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text     = Loader(tmp_path).load()
        chunks   = Chunker().chunker(text)
        embedder = Embedder()
        vectors  = embedder.embed(chunks)
        store    = VectorStorage(dimension=len(vectors[0]))
        store.add(vectors, chunks)
    finally:
        os.unlink(tmp_path)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"store": store, "embedder": embedder}

    return {"session_id": session_id, "message": "PDF indexed. Ready to chat!"}



class ChatRequest(BaseModel):
    session_id: str
    message:    str
    history:    list

@app.post("/chat")
async def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    store    = session["store"]
    embedder = session["embedder"]

    retriever      = Retriever(store, embedder, k=3)
    context_chunks = retriever.retrieve(req.message)

    if not context_chunks:
        return {"response": "I couldn't find relevant information in the document."}

    context_text  = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a research assistant.\n"
        "Format your answer STRICTLY like this:\n"
        "- Use numbered headings\n"
        "- Use bullet points under each heading\n"
        "- Add line breaks between sections\n"
        "- Keep spacing clean and readable\n"
        "DO NOT write everything in one paragraph."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(req.history)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {req.message}"})

    def token_stream():
    for model in MODELS:
        try:
            client = InferenceClient(model, token=os.environ["HF_TOKEN"])
            logger.info(f"Streaming with: {model}")
            success = False
            for token in client.chat_completion(messages, max_tokens=512, stream=True):
                text = token.choices[0].delta.content
                if text:
                    success = True
                    yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"
            return  
        except Exception as e:
            if success:
                
                yield "data: [DONE]\n\n"
                return
            logger.warning(f"Streaming failed for {model}: {e}")
            continue  

    
    yield "data: Sorry, all models are currently unavailable. Try again later.\n\n"
    yield "data: [DONE]\n\n"
            

    return StreamingResponse(token_stream(), media_type="text/event-stream")  


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
