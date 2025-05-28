import subprocess
import json

def generate_embedding(text: str):
    try:
        # Run ollama CLI with embedding flag and prompt as argument
        # Adjust according to ollama CLI docs if needed
        result = subprocess.run(
            ["ollama", "run", "llama3", "--embedding"],
            input=text,
            capture_output=True,
            text=True,
            check=True
        )
        embedding_json = json.loads(result.stdout)
        # Example expected key: "embedding" or adapt if different
        return embedding_json.get("embedding", [])
    except Exception as e:
        print("Embedding generation error:", e)
        return []
