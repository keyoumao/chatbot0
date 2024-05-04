import openai
import asyncio

class Chatbot:
    def __init__(self):
        # Initialize any necessary properties here, if required
        pass

    async def chat(self, user_input: str) -> str:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            max_tokens=256,
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            stream=True
        )
        
        content = ""
        async for token in response:
            content += token.choices[0].delta.get("content", "")
            #print() for a more instant response.
        return content.strip()

    def chatiteration(self):
        loop = asyncio.get_event_loop()
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = loop.run_until_complete(self.chat(user_input))
            print(f"Bot: {response}")
# flask builiding.

# Make sure to setup your OpenAI API key before running
# openai.api_key = 'YOUR_API_KEY'

# Create an instance of Chatbot and run it
if __name__ == "__main__":
    bot1 = Chatbot()
    bot1.chatiteration()
