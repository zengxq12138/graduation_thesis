import os

from openai import OpenAI

client = OpenAI(
    # api_key=os.getenv("DMX"),
    api_key="sk-E5wWzyK17gGOaTgKLnE4EfbJrsJwXcIk5EJv7CVjeBZYwcqM",
    base_url="https://ai.aiistart.com/v1"
)

completion = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        # {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},
        {"role": "user", "content": "你是谁"}
    ],
    top_p=0.7,
    temperature=0.9
)

print(completion.choices[0].message.content)
