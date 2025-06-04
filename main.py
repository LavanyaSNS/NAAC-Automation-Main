from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from processor import download_from_s3_url, extract_text_from_file, chunk_text
from vector_store import store_in_vector_db
from search import search_documents
from config import embedding_model
import os
from datetime import datetime

app = FastAPI(title="NAAC Document API")


class ProcessRequest(BaseModel):
    s3_urls: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/process", summary="Process and store documents from S3 URLs")
def process_documents_api(request: ProcessRequest):
    responses = []
    for s3_url in request.s3_urls:
        try:
            local_path = download_from_s3_url(s3_url)
            filename = os.path.basename(local_path)
            text = extract_text_from_file(local_path)
            os.remove(local_path)

            if not text:
                responses.append({"s3_url": s3_url, "status": "No text extracted"})
                continue

            chunks = chunk_text(text)
            base_metadata = {
                "s3_url": s3_url,
                "filename": filename,
                "processing_date": datetime.now().isoformat(),
                "file_type": filename.split(".")[-1].lower(),
            }

            metadatas = [
                {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                for i in range(len(chunks))
            ]

            store_in_vector_db(chunks, metadatas)
            responses.append(
                {
                    "s3_url": s3_url,
                    "filename": filename,
                    "chunks_stored": len(chunks),
                    "status": "Processed successfully",
                }
            )
        except Exception as e:
            responses.append({"s3_url": s3_url, "status": f"Failed: {str(e)}"})

    return {"results": responses}


@app.post("/search", summary="Search documents by query")
def search_api(request: SearchRequest):
    results = search_documents(request.query, request.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No documents found matching the query.")
    return {"results": results}
