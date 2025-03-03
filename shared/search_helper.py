import re

import json
import logging
from shared.utils import extract_json_from_markdown


def generate_metadata_filter_advanced(
    first_query: str,
    metadata_prompt: str,
    metadata_keys: list[str],
    google_genai_client,
) -> dict:
    """
    Takes the user's first query variation, plus any metadata prompt & metadata keys
    for the project, then calls an LLM to produce a Pinecone metadata filter.

    If the LLM determines that no filter is needed, or if the LLM output is invalid,
    returns {} (an empty dict).

    Example usage:
        filter_dict = generate_metadata_filter_advanced(first_query, metadata_prompt, metadata_keys)

    Then pass 'filter=filter_dict' to your Pinecone .query(...) call, if filter_dict is not empty.
    """

    # (1) Build an advanced prompt with details about Pinecone’s filter syntax
    # and instructions on how to produce a valid JSON object.
    # We reference the Pinecone doc: https://docs.pinecone.io/docs/metadata-filtering
    # about the operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $and, $or.
    # Also incorporate your project’s metadata_prompt if it helps the LLM figure out
    # the domain or how to map user query to metadata fields.

    system_instruction = f"""
You are a highly capable AI system with knowledge of the Pinecone metadata filtering syntax.

Below is relevant information about Pinecone filter operators:
- $eq  : field equals value (numbers, strings, bool)
- $ne  : field not equal to value
- $gt  : field greater than some numeric value
- $gte : field greater or equal
- $lt  : field less than numeric value
- $lte : field less or equal
- $in  : field is in array of possible values
- $nin : field is not in array
- $exists : checks if field exists (true/false)
- $and/$or : logical chaining

Your job is to produce a single valid JSON object that can be used as a Pinecone filter.

We also have a set of potential metadata keys for this project. You are to only choose possible key names from below list if a possible value if found in user query in `QUERY# setion :
{metadata_keys}

Additionally, here is context around the topic user is querying around:
\"\"\"{metadata_prompt}\"\"\"

QUERY#:
\"\"\"{first_query}\"\"\"

ALWAYS create a list of variation of values when key type is `String` or `List` in metadata Keys list above.
ALWAYS create atleast 10 variation of original value picked, add abbreviations, jumbled words, synonyms, address variations like Street, st., str. so that possibleity of matching tags is high.
For example: 
 - If value found for key `Country` is `usa` then create a list of values like [`usa`, `united states`, `united states of america`, `us of a`]
 - If value found for `address` key is `111 some renadom street` then create a list of values like [`111 some renadom street`, `111 some renadom st`, `111 some renadom, and so on`]
 - use $in filter as much possible unless it's matching date or unique numbers. use your common sense and intelligence as much possible.
From the user’s query, decide which of these metadata fields (keys) might be relevant to filter by.
If the user query does not mention anything that matches or references those metadata fields,
OR you cannot confidently build a filter, then respond with an empty JSON object {{}}.

If you do create a filter, you must strictly return valid JSON representing that filter, e.g.:
{{ "fieldName": {{"$in": ["someValue1", "somevalue2"]}} }}

If multiple fields apply, you can chain them with $and or $or. For example:
{{"$and": [
    {{"region": {{"$eq": "APAC"}}}},
    {{"year": {{"$gte": 2020}}}}
]}}
But do not add extra keys or textual explanation. The result must be pure JSON.

Important:
- Return NOTHING except the filter JSON. No code fences, no extra text.
- If no filter is relevant, return {{}}
"""

    # (2) Call your LLM with the above prompt.
    # Example with google_genai_client:
    try:
        response = google_genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Or whichever model
            contents=system_instruction,
            config={"temperature": 0.0},
        )
        raw_output = extract_json_from_markdown(response.text.strip())

        logging.info(f'Raw metadata Filter created {raw_output}')
        # (3) Attempt to parse as JSON
        try:
            filter_obj = json.loads(raw_output)
            # Validate we got a dict
            if not isinstance(filter_obj, dict):
                logging.info(
                    "generate_metadata_filter_advanced => LLM did not return a JSON object, returning {}"
                )
                return {}
            # Return the filter
            return filter_obj
        except json.JSONDecodeError:
            logging.info(
                "generate_metadata_filter_advanced => Could not parse LLM output as JSON."
            )
            return {}

    except Exception as e:
        logging.error(f"generate_metadata_filter_advanced => LLM call failed: {e}")
        return {}


def create_multi_query_prompt(history, query):
    """
    Prompt to generate multiple queries, including variations based on the current query,
    one variation considering history, and one keyword-only query.
    """
    # This is similar to what you had in the second snippet, tailored for your environment.
    if history and len(history) > 0:
        return f"""You are an AI language model assistant. Your task is
to generate 2 different versions of the given user question to retrieve
relevant documents from a vector database. Generate 1 variation based on
the current question and considering the conversation
history provided below, and create 1 variation of original question by creating
a list of just keywords by picking names, addresses, regions, and similar unique values from user query.
DO NOT pick verbs or adjectives as keywords. Be very selective in choosing these keywords.
The response MUST only have generated queries separated by newlines.
Do not number or bullet them. DO NOT add anything before and after generated queries.
Put the keyword-only query last.

Conversation history:
{history}

Current question:
{query}
"""
    else:
        return f"""You are an AI language model assistant tasked with generating two distinct query variations to improve document retrieval from a vector database.

1. Return the original query as is without change.
2. Keyword Extraction: Identify and extract only unique identifiers (such as names, addresses, regions, or similar unique values) from the original question. Do not include verbs or adjectives. Output these unique keywords as a JSON array.

Output Requirements:
- Provide exactly two generated queries separated by a newline.
- The first line must be the rephrased query.
- The second line must be the keyword-only query in a JSON array.
- Do not include any extra text, numbering, or bullet points.
- Ensure that the keyword-only query appears as the final line.

Examples:
If the current question is: "Find the best restaurants in New York near Central Park", acceptable outputs would be:

Returned original query:
"Find the best restaurants in New York near Central Park"

Keyword Array:
["New York", "Central Park"]

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
        doc_text = doc["text"] if "text" in doc else doc.get("page_content", "")
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
