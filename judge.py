import os, json, time
from google import genai
from google.genai import types, errors

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

PROMPT = """You are judging music search results for rappers/singers looking for a beat.

Query: "{query}"
Track tags: {tags}

Score how well this track fits the query:
2 = strong match: mood AND genre/instruments fit the query
1 = partial match: mood OR genre fits, not both
0 = no match

Return JSON only: {{"score": 0|1|2, "reason": "<one short sentence>"}}"""

def judge(query, tags, max_tries=5):
    for attempt in range(max_tries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=PROMPT.format(query=query, tags=", ".join(tags)),
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(resp.text)
            return data["score"], data["reason"]
        except errors.ServerError:
            wait = 2 ** attempt          # 1, 2, 4, 8, 16 seconds
            print(f"Server busy, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Judge failed after retries")

if __name__ == "__main__":
    q = "dark moody trap beat"
    print(judge(q, ["dark", "hiphop", "synthesizer"]))
    print(judge(q, ["dark", "classical", "piano"]))
    print(judge(q, ["happy", "pop", "ukulele"]))
