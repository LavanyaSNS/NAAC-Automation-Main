from typing import List, Dict

from config import collection, embedding_model, s3_client, PRESIGNED_URL_EXPIRATION


def generate_presigned_url(s3_url: str, expiration: int = PRESIGNED_URL_EXPIRATION) -> str:
    import re

    pattern = r"s3://([^/]+)/(.+)"
    match = re.match(pattern, s3_url)
    if not match:
        return ""
    bucket, key = match.group(1), match.group(2)
    try:
        url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiration
        )
        return url
    except Exception as e:
        print(f"Error generating presigned URL: {str(e)}")
        return ""


def search_documents(user_query: str, top_k: int = 5) -> List[Dict]:
    try:
        query_embedding = embedding_model.encode(user_query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        document_results = {}
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            doc_url = metadata["s3_url"]
            distance = results["distances"][0][i]
            score = 1 - distance

            if doc_url not in document_results:
                document_results[doc_url] = {
                    "s3_url": doc_url,
                    "filename": metadata["filename"],
                    "file_type": metadata["file_type"],
                    "max_score": score,
                    "best_chunk": results["documents"][0][i],
                }
            elif score > document_results[doc_url]["max_score"]:
                document_results[doc_url]["max_score"] = score
                document_results[doc_url]["best_chunk"] = results["documents"][0][i]

        sorted_results = sorted(
            document_results.values(), key=lambda x: x["max_score"], reverse=True
        )

        return [
            {
                "document_name": res["filename"],
                "document_type": res["file_type"],
                "relevance_score": res["max_score"],
                "s3_url": res["s3_url"],
                "download_url": generate_presigned_url(res["s3_url"]),
                "matching_excerpt": (res["best_chunk"][:500] + "...")
                if len(res["best_chunk"]) > 500
                else res["best_chunk"],
            }
            for res in sorted_results
        ]

    except Exception as e:
        print(f"Error searching documents: {str(e)}")
        return []
