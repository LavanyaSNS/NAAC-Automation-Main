import io
import fitz  # PyMuPDF
import docx
import pandas as pd

def extract_text_from_file(filename, file_bytes):
    extension = filename.split(".")[-1].lower()
    if extension == "pdf":
        return extract_pdf_text(file_bytes)
    elif extension == "docx":
        return extract_docx_text(file_bytes)
    elif extension in ["xls", "xlsx"]:
        return extract_excel_text(file_bytes)
    else:
        return ""

def extract_pdf_text(file_bytes):
    text = ""
    with fitz.open("pdf", file_bytes) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_docx_text(file_bytes):
    text = ""
    file_stream = io.BytesIO(file_bytes)
    doc = docx.Document(file_stream)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_excel_text(file_bytes):
    file_stream = io.BytesIO(file_bytes)
    df = pd.read_excel(file_stream, engine='openpyxl')
    return df.to_string()
