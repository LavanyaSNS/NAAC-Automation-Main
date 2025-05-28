def store_document(collection, filename, text, embedding):
    doc = {
        "filename": filename,
        "text": text,
        "embedding": embedding
    }
    result = collection.insert_one(doc)
    return result.inserted_id

def get_all_document_embeddings(collection):
    docs = collection.find({}, {"embedding": 1, "filename": 1})
    return [{"id": str(doc["_id"]), "embedding": doc["embedding"], "filename": doc["filename"]} for doc in docs]
