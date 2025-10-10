import json
import re

def parse_llm_json(result_text: str):
    """
    Safely extract and parse JSON from LLM outputs that may include Markdown code fences.
    """
    if not result_text or not isinstance(result_text, str):
        return {"title": "No title found", "article": "No article found"}

    # 1️ Remove Markdown fences like ```json ... ```
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", result_text.strip())

    # 2️ Try parsing JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 3️ Fallback: extract JSON substring if the model added extra text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # 4️ Last resort: return plain text
        return {"title": "No title found", "article": cleaned.strip()}