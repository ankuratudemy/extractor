
import re


def extract_json_from_markdown(text: str) -> str:
    """
    If the text is wrapped in markdown code fences (e.g. ```json ... ```),
    extract and return only the inner JSON. Otherwise, return the original text.
    """
    # This regex matches ```json at the start, then captures everything until the closing ```
    pattern = r"^```(?:json)?\s*(\{.*\})\s*```$"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text