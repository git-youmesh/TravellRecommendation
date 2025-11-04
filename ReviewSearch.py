import pandas as pd
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import scipy.spatial
import faiss
import time
dataset = load_dataset("traversaal-ai-hackathon/hotel_datasets")
df=pd.DataFrame(dataset['train'])
df_paris = df.loc[df.locality=='Paris']
model = SentenceTransformer("all-MiniLM-L6-v2")
if torch.cuda.is_available():
    model = model.to('cuda')
    print("CUDA is available. The model has been moved to GPU.")
else:
    print("CUDA is not available. The model will run on CPU.")
reviews = df_paris['review_text'].tolist()
query = "Hotel near the Louvre with great food nearby."
review_embeddings = model.encode(reviews, show_progress_bar=True)
query_embedding = model.encode([query]).astype('float32')
review_embeddings = review_embeddings.astype('float32')
review_embeddings_normalized = review_embeddings / np.linalg.norm(review_embeddings)
index = faiss.IndexFlatIP(review_embeddings_normalized.shape[1])
index.add(review_embeddings_normalized)
query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding) 
k = 5 
start_time = time.time()
distances, indices = index.search(query_embedding_normalized, k)
faiss_search_time = time.time() - start_time
print(f"Faiss search time: {faiss_search_time:.4f} seconds")
print(f"Query: {query}")
print("Top hotel with similar reviews using FAISS:")
for i, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"{i}. {df_paris.iloc[idx]['hotel_name']}")
    print(f"Review: {df_paris.iloc[idx]['review_text']}")
    print(f"Distance: {distance:.4f}")
 
