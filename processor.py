import os
import re
import tempfile
from datetime import datetime
from typing import List

import pandas as pd
import PyPDF2
from docx import Document

from config import s3_client

TEMP_DIR = tempfile.gettempdir()
CHUNK_SIZE = 1000  # words


def parse_s3_url(s3_url: str):
    pattern = r"s3://([^/]+)/(.+)"
    match = re.match(pattern, s3_url)
    if not match:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")
    return {"bucket": match.group(1), "key": match.group(2)}


def download_from_s3_url(s3_url: str) -> str:
    s3_info = parse_s3_url(s3_url)
    filename = os.path.basename(s3_info["key"])
    local_path = os.path.join(TEMP_DIR, filename)

    s3_client.download_file(
        Bucket=s3_info["bucket"],
        Key=s3_info["key"],
        Filename=local_path,
    )
    return local_path


def extract_text_from_file(file_path: str) -> str:
    try:
        ext = file_path.lower().split(".")[-1]
        if ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(
                    [page.extract_text() or "" for page in reader.pages]
                )
        elif ext in ("docx", "doc"):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text])
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_path)
            return df.to_string()
        else:
            print(f"Unsupported file type: {file_path}")
            return ""
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
