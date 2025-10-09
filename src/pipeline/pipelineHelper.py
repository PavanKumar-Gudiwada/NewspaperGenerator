

def format_rag_response(response: dict) -> str:
    """
    Format a RAG response dictionary for clean display.
    Removes emojis and paragraph numbering.
    """
    query = response.get("query", "N/A")
    result_text = response.get("result", "").strip()

    # Split lines and extract title + paragraphs
    lines = [line.strip() for line in result_text.split("\n") if line.strip()]
    title = ""
    paragraphs = []

    for line in lines:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip().strip('"')
        elif line.lower().startswith("paragraph"):
            paragraphs.append(line.split(":", 1)[1].strip())
        else:
            paragraphs.append(line)

    # Build formatted display
    parts = []
    if title:
        parts.append(f"Title: {title}\n")

    for para in paragraphs:
        parts.append(f"{para}\n")

    return "\n".join(parts).strip()
