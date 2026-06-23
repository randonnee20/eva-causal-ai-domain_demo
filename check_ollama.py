import requests

r = requests.get("http://localhost:11434/api/tags", timeout=3)
models = r.json().get("models", [])
print(f"설치된 모델 수: {len(models)}\n")
for m in models:
    name = m["name"]
    size_gb = round(m.get("size", 0) / 1e9, 1)
    print(f"  {name:<40}  {size_gb} GB")
