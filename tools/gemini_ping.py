import os
import sys
import time
from dotenv import load_dotenv

load_dotenv(r"C:\Users\arman\Projects\harmony_vlm\.env")
key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

from google import genai

client = genai.Client(api_key=key)
models = sys.argv[1:] or ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-flash-latest"]
for m in models:
    t0 = time.time()
    try:
        r = client.models.generate_content(model=m, contents="Reply with the single word OK.")
        print(f"{m:30s} OK   {(time.time()-t0)*1000:5.0f}ms  {(r.text or '').strip()[:30]}")
    except Exception as e:
        print(f"{m:30s} FAIL {(time.time()-t0)*1000:5.0f}ms  {type(e).__name__}: {str(e)[:100]}")
