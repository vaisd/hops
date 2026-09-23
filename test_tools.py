import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["SUBCONSCIOUS_API_KEY"],
    base_url="https://api.subconscious.dev/v1",
)

try:
    r = client.chat.completions.create(
        model="subconscious/deepseek-v4.1-flash-marathon",
        messages=[{"role": "user", "content": "What's the weather in Boston?"}],
        tools=[{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }}],
    )
    print(json.dumps(r.model_dump(), indent=2))
except Exception as e:
    print("ERROR:", type(e).__name__, e)