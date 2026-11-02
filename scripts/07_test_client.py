"""Step 8: call the vLLM server with the OpenAI Python client (pip install openai)."""
import argparse
import os

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8000/v1")
parser.add_argument("--model", default="my-lora", help="adapter name (lora mode) or served name")
parser.add_argument("--prompt", default="Explain what LoRA fine-tuning is in two sentences.")
parser.add_argument("--stream", action="store_true")
args = parser.parse_args()

client = OpenAI(base_url=args.base_url, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"))

print("Available models:", [m.id for m in client.models.list().data])

resp = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": args.prompt}],
    max_tokens=300,
    temperature=0.7,
    stream=args.stream,
)
if args.stream:
    for chunk in resp:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()
else:
    print(resp.choices[0].message.content)
    print("Usage:", resp.usage)
