import re

def validate_document_format(text):
    if not re.search(r"\b\d{2}-\d{2}-\d{4}\b", text):
        return {"is_valid": False, "reason": "Missing date in DD-MM-YYYY format"}

    required_sections = ["Objectives", "Outcome", "Methodology"]
    missing = [sec for sec in required_sections if sec.lower() not in text.lower()]
    if missing:
        return {"is_valid": False, "reason": f"Missing sections: {', '.join(missing)}"}

    return {"is_valid": True}
