import os
import boto3
import tempfile
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = "bmc-automation"
FOLDER_NAME = "naac_documents"

s3 = boto3.client(
"s3",
aws_access_key_id=AWS_ACCESS_KEY,
aws_secret_access_ke,m  y=AWS_SECRET_KEY,
region_name="ap-south-1"
)

model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
doc_map = [] # Stores metadata

def list_s3_objects():
response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_NAME)
files = response.get("Contents", [])
return [file["Key"] for file in files if file["Key"].endswith(".pdf")]

def download_pdf_from_s3(key: str) -> str:
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
s3.download_file(BUCKET_NAME, key, tmp.name)
return tmp.name

def extract_text_from_pdf(path: str) -> str:
with pdfplumber.open(path) as pdf:
text = "\n".join(page.extract_text() or "" for page in pdf.pages)
return text.strip()

def embed_text(text: str) -> np.ndarray:
return model.encode([text])[0]

def process_documents():
keys = list_s3_objects()
for key in keys:
try:                                                                      
path = download_pdf_from_s3(key)
text = extract_text_from_pdf(path)
embedding = embed_text(text)
index.add(np.array([embedding]))
doc_map.append({
"s3_key": key,
"embedding": embedding,
"text": text[:1000] + "...", # short preview
"url": f"https://{BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{key}"
})
except Exception as e:
print(f"Error processing {key}: {e}")

def search_documents(query: str, top_k: int = 5) -> List[dict]:
q_vec = embed_text(query)
D, I = index.search(np.array([q_vec]), top_k)
return [doc_map[i] for i in I[0] if i < len(doc_map)]