from sentence_transformers import SentenceTransformer

# small fast pretrained model (downloads once, ~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# sample words 
words = ["king", "queen", "prince", "apple", "banana", "mango", "car", "truck", "bicycle"]

embeddings = model.encode(words)


print(f"Number of words: {len(words)}")
print(f"Shape of embeddings: {embeddings.shape}")  
print(f"\nFirst 10 numbers for the word 'king':")
print(embeddings[0][:10])