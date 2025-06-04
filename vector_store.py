import uuid
from typing import List, Dict

from config import collection, embedding_model


def store_in_vector_db(texts: List[str], metadatas: List[Dict]):
    try:
        embeddings = embedding_model.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in texts]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
    except Exception as e:
        print(f"Error storing in Chroma DB: {str(e)}")

