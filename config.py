import os
import boto3
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()  # loads .env file variables

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
PRESIGNED_URL_EXPIRATION = 3600  # 1 hour

# AWS S3 Setup
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-south-1"
)
BUCKET_NAME = "bmc-automation"
# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize Chroma DB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Get or create collection
collection = chroma_client.get_or_create_collection(name="naac_documents")
