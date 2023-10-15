import openai

result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",messages=[
        {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Create an blog article"},
         ]
)
    
print(result.choices[0].text)
