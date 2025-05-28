from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from utils.text_extraction import extract_text_from_file
from utils.embedding import generate_embedding
from utils.vector_search import search_similar_documents
from utils.storage import store_document, get_all_document_embeddings
from utils.validation import validate_document_format

app = FastAPI()
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["NAAC_OKRion"]
collection = db["test"]

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_file(file.filename, contents)
    if not text:
        return JSONResponse(content={"error": "Unsupported file or empty content"}, status_code=400)

    validation_result = validate_document_format(text)
    if not validation_result["is_valid"]:
        return {"status": "Rejected", "reason": validation_result["reason"]}

    embedding = generate_embedding(text)
    doc_id = store_document(collection, file.filename, text, embedding)
    return {"status": "Stored", "doc_id": str(doc_id)}

@app.get("/search/")
def search_documents(query: str):
    query_embedding = generate_embedding(query)
    all_docs = get_all_document_embeddings(collection)
    matches = search_similar_documents(query_embedding, all_docs)
    return {"matches": matches}
