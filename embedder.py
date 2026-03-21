import os
import requests
 
class Embedder:
    """
    Converts text into vectors using HuggingFace Inference API.
    No local model loading — zero heavy dependencies.
    """
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
 
    def _get_embeddings(self, texts):
        """Call HuggingFace API and return list of vectors."""
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "inputs": texts,
                "options": {"wait_for_model": True}  # wait if model is cold
            }
        )
        response.raise_for_status()
        return response.json()
 
    def embed(self, chunks):
        """Embed a list of text chunks — same interface as before."""
        return self._get_embeddings(chunks)
 
    def embed_q(self, query):
        """Embed a single query string — same interface as before."""
        result = self._get_embeddings([query])
        return result[0]   # unwrap the single vector
