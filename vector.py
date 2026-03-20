import faiss
import numpy as np

class VectorStorage:
    def __init__(self, dimension=384): 
    
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add(self, vectors, chunks):
        
        v_array = np.array(vectors).astype('float32')
        
        faiss.normalize_L2(v_array)
        
        self.index.add(v_array)
        self.chunks.extend(chunks)

    def search(self, query_vector, k=5):
        
        q_array = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(q_array)
        
        
        scores, indices = self.index.search(q_array, k)
        return scores[0], indices[0]