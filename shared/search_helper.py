import re

def create_multi_query_prompt(history, query):
    """
    Prompt to generate multiple queries, including variations based on the current query,
    one variation considering history, and one keyword-only query.
    """
    # This is similar to what you had in the second snippet, tailored for your environment.
    if history and len(history) > 0:
        return f"""You are an AI language model assistant. Your task is
to generate 5 different versions of the given user question to retrieve
relevant documents from a vector database. Generate 3 variations based on
the current question, 1 variation considering the conversation
history provided below, and create 1 variation of original question by creating just keywords by picking names, addresses, regions. DO NOT pick verbs or adjectives as keywords.
The response MUST only have generated queries separated by newlines.
Do not number or bullet them. DO NOT add anything before and after generated queries.
Put the keyword-only query last.

Conversation history:
{history}

Current question:
{query}
"""
    else:
        return f"""You are an AI language model assistant. Your task is
to generate 3 different versions of the given user question to retrieve
relevant documents from a vector database. Generate 2 variations based on
the current question and create 1 variation of the original question as a keyword-only query by creating just keywords by picking names, addresses, regions. DO NOT pick verbs or adjectives as keywords.
Do not number or bullet them. DO NOT add anything before and after generated queries.
Put the keyword-only query last.

Current question:
{query}
"""


async def reorder_docs(docs, keywords):
    """
    Reorder docs by placing those containing any of the given keywords at the top.
    """
    matches = []
    non_matches = []

    for doc in docs:
        matched = False
        doc_text = doc['text'] if 'text' in doc else doc.get(
            'page_content', '')
        for kw in keywords:
            kw_clean = kw.strip()
            if re.search(r"\b" + re.escape(kw_clean) + r"\b", doc_text, re.IGNORECASE):
                matches.append(doc)
                matched = True
                break
        if not matched:
            non_matches.append(doc)

    # Limit matches to top 20 (arbitrary, can adjust)
    matches = matches[:20]
    return matches + non_matches



