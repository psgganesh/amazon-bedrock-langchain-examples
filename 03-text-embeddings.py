from langchain_aws import BedrockEmbeddings

# Initialize the Bedrock Embeddings model (using Titan)
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

# Generate embeddings for a single text
text = "This is a sample text for embedding."
embedding = embeddings.embed_query(text)

print(f"Embedding for '{text}':")
print(f"Dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Generate embeddings for multiple texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a versatile programming language."
]

embeddings_list = embeddings.embed_documents(texts)

print("\nEmbeddings for multiple texts:")
for i, emb in enumerate(embeddings_list):
    print(f"Text {i+1} - Dimension: {len(emb)}, First 5 values: {emb[:5]}")