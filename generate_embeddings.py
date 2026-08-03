from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.decomposition import PCA



# small fast pretrained model (downloads once, ~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# sample words 
words = ["king", "queen", "prince", "apple", "banana", "mango", "car", "truck", "bicycle"]

embeddings = model.encode(words)


print(f"Number of words: {len(words)}")
print(f"Shape of embeddings: {embeddings.shape}")  
print(f"\nFirst 10 numbers for the word 'king':")
print(embeddings[0][:10])

# Compressing 384 dimensions down to 2 dimensions
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Prints the 2D version for each word
for word, coord in zip(words, embeddings_2d):
    print(f"{word}: {coord}")

plt.figure(figsize=(9, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=60, color='#1f77b4', zorder=2)

# Creates text labels first, then lets the adjustText reposition them to avoid overlap
texts = []
for word, coord in zip(words, embeddings_2d):
    texts.append(plt.text(coord[0], coord[1], word, fontsize=13))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

plt.title("Word Meanings, Visualized", fontsize=14)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True, alpha=0.3)
plt.savefig("embedding_plot.png", dpi=150, bbox_inches='tight')
plt.show()