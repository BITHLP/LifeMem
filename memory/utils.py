from openai import OpenAI

openai_key=""

def openai_chat(prompt, model):
    client = OpenAI(api_key=openai_key)
    try:
        if "gpt-5" in model:
            completion = client.chat.completions.create(model=model, messages=prompt, stream=False, reasoning_effort="low")
        else:
            completion = client.chat.completions.create(model=model, messages=prompt, stream=False)
        return completion.choices[0].message.content
    except Exception as e:
        print("Exception: ", e)
        return "OpenAI API error"
def get_response(prompt, model="gpt-5-mini-2025-08-07"):
    retry=0
    while(retry<5):
        prompt=[{"role": "user", "content": prompt}]
        response = openai_chat(prompt, model)
        if response != "OpenAI API error":
            break
        else:
            retry+=1
    return response