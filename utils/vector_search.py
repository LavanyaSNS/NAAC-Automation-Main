from sklearn.metrics.pairwise import cosine_similarity

def search_similar_documents(query_embedding, documents, top_k=5):
    if not query_embedding:
        return []

    results = []
    for doc in documents:
        sim = cosine_similarity(
            [query_embedding], [doc["embedding"]])[0][0]
        results.append({"filename": doc["filename"], "score": round(float(sim), 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
