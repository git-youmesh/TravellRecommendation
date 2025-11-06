#Text classification steps 
"""Convert the input documents to embeddings with an embedding
model.
2. Reduce the dimensionality of embeddings with a dimensionality
reduction model.
3. Find groups of semantically similar documents with a cluster
model.
"""


from datasets import load_dataset
from transformers import pipeline
from huggingface_hub import login
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from umap import UMAP
from hdbscan import HDBSCAN
mytoken =" "
login(token=mytoken)
def evaluate_performance(y_true, y_pred):
  performance = classification_report(
  y_true, y_pred,
  target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

dataset = load_dataset("maartengr/arxiv_nlp")["train"]
# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Create an embedding for each abstract
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts,
show_progress_bar=True)



# We reduce the input embeddings from 384 dimensions to 5

umap_model = UMAP(
 n_components=5, min_dist=0.0, metric='cosine',
random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# We fit the model and extract the clusters
hdbscan_model = HDBSCAN(
 min_cluster_size=50, metric="euclidean",
cluster_selection_method="eom"
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_
cluster = 0
for index in np.where(clusters==cluster)[0][:3]:
 print(abstracts[index][:300] + "... \n")






