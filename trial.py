import time
import openai

def chat_with_gpt3(prompt, retries=3, delay=5):
    for _ in range(retries):
        try:
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except openai.error.APIConnectionError:
            time.sleep(delay)  # Wait for a few seconds before retrying
    raise Exception("Failed to get a response after multiple retries")

print(chat_with_gpt3("You are a helpful assistant."))
