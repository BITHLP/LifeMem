from openai import OpenAI
import requests

openai_key=""

openai_chat_model = "gpt-4o-mini-2024-07-18"

def process(result):
    if "<" in result and ">" in result:
        id1=result.find("<")
        id2=result.find(">")
        return result[id1+1:id2]
    else:
        return result

def openai_chat(prompt, model):
    client = OpenAI(api_key=openai_key)
    try:
        completion = client.chat.completions.create(model=model, messages=prompt, temperature=0, stream=False)
        result=completion.choices[0]
        result=process(result.message.content)
        return result
    except Exception as e:
        print(e)
        return "OpenAI API error"

def get_response(prompt, model=openai_chat_model):
    for i in range(5):
        response = openai_chat(prompt, model).replace("Assistant: ","")
        if response != "OpenAI API error" and response != "Local API error":
            break
    return response