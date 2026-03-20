class Retriever:
    def __init__(self, vector_store, embedder, k=5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.k = k

    def retrieve(self, query):
        
        vquery = self.embedder.embed_q(query)
        
        
        scores, indices = self.vector_store.search(vquery, self.k)

        
        if scores[0] < 0.5:
            print(f"Evidence too low: {scores[0]}")
            return []

        
        results = [
            self.vector_store.chunks[i] 
            for i in indices if i != -1
        ]
        return results




