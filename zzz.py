from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v1",  # 指向vLLM服务
  api_key="dummy"  # vLLM无需实际API密钥，填任意值即可
)

response = client.chat.completions.create(
  model="Qwen3-32B-AWQ",
  messages=[{"role": "user", "content": "什么是vLLM？"}]
)

print(response.choices[0].message.content)