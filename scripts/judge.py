import csv, os, time, json
from google import genai
from google.genai import types, errors

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
OUT = 'eval/labels.csv'
done = set()
if os.path.exists(OUT):
    done = {(r['prompt_id'], r['track_id']) for r in csv.DictReader(open(OUT))}

RUBRIC = """You are judging a music search engine for rappers/singers looking for an instrumental beat.
Search prompt: "{prompt}"
Listen to the clip and grade how well it matches the prompt's vibe:
2 = nails it, 1 = partly matches, 0 = doesn't match.
Reply as JSON: {{"grade": 0|1|2, "reason": "<one short sentence>"}}"""

def judge(audio, prompt):
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[types.Part.from_bytes(data=audio, mime_type='audio/mp3'),
                          RUBRIC.format(prompt=prompt)],
                config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0))
            return json.loads(resp.text)
        except (errors.ServerError, errors.ClientError) as e:
            if getattr(e, 'code', None) not in (429, 500, 503): raise
            wait = 15 * (attempt + 1)
            print(f'  busy ({e.code}), retrying in {wait}s')
            time.sleep(wait)
    raise RuntimeError('still busy after 6 tries, re-run later')

new = not os.path.exists(OUT)
with open(OUT, 'a', newline='') as f:
    w = csv.writer(f)
    if new: w.writerow(['prompt_id', 'track_id', 'grade', 'reason'])
    for r in csv.DictReader(open('eval/pool.csv')):
        if (r['prompt_id'], r['track_id']) in done: continue
        audio = open(f"data/clips/{r['path']}", 'rb').read()
        j = judge(audio, r['prompt'])
        w.writerow([r['prompt_id'], r['track_id'], j['grade'], j['reason']]); f.flush()
        print(r['prompt_id'], r['track_id'], j['grade'], j['reason'])
        time.sleep(7)   # stay under free-tier rate limit
