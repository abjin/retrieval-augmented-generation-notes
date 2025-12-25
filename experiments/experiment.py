import os
import time
from pinecone import Pinecone
from google.genai import types
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("rag-notes")

start_time = time.time()

prompt = "쿠버네티스가 무엇인가요?"
response = client.models.generate_content_stream(
    model="gemini-3-flash-preview",
    contents=prompt
)


ttft = None
last_chunk = None
result = ""

for chunk in response:
    if ttft is None:
        ttft = time.time() - start_time
    result += chunk.text
    last_chunk = chunk

end_time = time.time()

print({
    "prompt_tokens": last_chunk.usage_metadata.prompt_token_count,
    "completion_tokens": last_chunk.usage_metadata.candidates_token_count,
    "total_tokens": last_chunk.usage_metadata.total_token_count,
    "ttft": ttft,
    "latency": end_time - start_time,
    "response": result
})
