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




text     = Loader("portfolio.pdf").load()
chunks   = Chunker().chunker(text)
embedder = Embedder()
vectors  = embedder.embed(chunks)
store    = VectorStorage(dimension=len(vectors[0]))
store.add(vectors, chunks)
   



class ChatRequest(BaseModel):
    session_id: str
    message:    str
    history:    list

@app.post("/chat")
async def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        sessions[req.session_id] = {"store": store, "embedder": embedder}
        session = sessions[req.session_id]


    store    = session["store"]
    embedder = session["embedder"]

    retriever      = Retriever(store, embedder, k=3)
    context_chunks = retriever.retrieve(req.message)

    if not context_chunks:
        return {"response": "I only answer questions about Aarav and his work."}

    context_text  = "\n\n".join(context_chunks)
    system_prompt = (
        "You are Aarav's AI assistant.\n"
        "Your name is Zooba\n"
        "Your job is to answer questions about Aarav Kumar Ranjan, his projects, skills, and interests using the provided context.\n"
        "Rules:\n"
        "- Only answer using the given context. Do not make up information.\n"
        "- If the answer is not in the context, say: I only answer questions about Aarav and his work.\n"
        "- Keep answers clear, simple, and confident.\n"
        "- Do not use complex jargon unless necessary.\n"
        "- Prefer explaining things in a way a beginner can understand.\n"
        "Style:\n"
        "- Speak in a calm, intelligent, and slightly friendly tone.\n"
        "- Be concise but informative.\n"
        "- When explaining projects, include:\n"
        " • what it does\n"  
        " • how it works (simple explanation)\n"  
        " • why it is useful\n" 
        "Do not generate fake achievements, skills, or experiences.\n" 
        "Do not pretend to be Aarav himself.\n"
        "If asked about projects, mention their names clearly.\n"
        "Make Aarav appear as a thoughtful, skilled, and curious machine learning enthusiast who focuses on understanding and building real systems.\n"
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

    
        yield "data: Sorry, we are currently unavailable. Try again later.\n\n"
        yield "data: [DONE]\n\n"
            

    return StreamingResponse(token_stream(), media_type="text/event-stream")  


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
