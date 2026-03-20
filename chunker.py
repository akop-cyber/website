class Chunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunker(self, text):
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        chunks = []
        start = 0
        text_size = len(text)

        while start < text_size:
            end = start + self.chunk_size
            
            
            if end < text_size:
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space 
            

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            
            start = end - self.overlap
            
        return chunks