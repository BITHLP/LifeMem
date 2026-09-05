from openai import OpenAI
import requests

openai_key=""

openai_chat_model = "gpt-4o-mini-2024-07-18"

def openai_chat(prompt, model):
    print(model)
    client = OpenAI(api_key=openai_key)
    try:
        completion = client.chat.completions.create(model=model, messages=prompt, temperature=0.3, max_tokens=512, stream=False, stop=["Observation"])
        return completion.choices[0].message.content
        
    except Exception as e:
        print(e)
        return "OpenAI API error"

def get_response(prompt, model=openai_chat_model):
    for i in range(5):
        response = openai_chat(prompt, model)
        if response != "OpenAI API error":
            break
    return response