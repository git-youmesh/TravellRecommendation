from datasets import load_dataset
from transformers import pipeline
from huggingface_hub import login
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
mytoken =""
login(token=mytoken)
def evaluate_performance(y_true, y_pred):
  performance = classification_report(
  y_true, y_pred,
  target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

# Load our data
data = load_dataset("rotten_tomatoes")

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"],
show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"],
show_progress_bar=True)
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])
y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)


label_embeddings = model.encode(["A negative review", "A positive review"])
