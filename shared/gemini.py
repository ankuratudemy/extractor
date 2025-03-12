from google import genai
from google.genai import types
import os
import asyncio

async def generate(chunk: str, topics_prompt: str, retries: int = 3):
    regions = [
        "us-central1",
        "us-east5",       # Columbus, Ohio
        "us-south1",      # Dallas, Texas
        "us-west4",       # Las Vegas, Nevada
        "us-east1",       # Moncks Corner, South Carolina
        "us-east4",       # Northern Virginia
        "us-west1"        # Oregon
    ]

    prompt_text = f"""{topics_prompt}\n\nText: '{chunk}'"""

    text1 = types.Part.from_text(text=prompt_text)

    model = "gemini-2.0-flash-lite-001"
    contents = [types.Content(role="user", parts=[text1])]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        response_mime_type="application/json",
        response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
    )

    for region in regions:
        client = genai.Client(
            vertexai=True,
            project=os.environ.get("GCP_PROJECT_ID"),
            location=region,
        )

        attempt = 0
        while attempt < retries:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                return response.text
            except Exception as e:
                error_message = str(e)
                print(f"Attempt {attempt + 1} in region {region} failed: {error_message}")
                if "429" in error_message:
                    print(f"Region {region} rate limited, switching to next region.")
                    break  # Break inner loop, move to next region
                attempt += 1
                await asyncio.sleep(2 ** attempt)

    print("All retry attempts in all regions failed. Returning empty array.")
    return []

# Example usage:
# asyncio.run(generate("Your dynamic text chunk here", "Your topics prompt here"))
