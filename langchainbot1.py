from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder, 
                               SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import (ConversationSummaryMemory, ConversationSummaryBufferMemory, 
                             ConversationBufferMemory)

class LangChainChatbot:

    def __init__(self):
        # LLM Initialization
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo",streaming=True, 
                              callbacks=[StreamingStdOutCallbackHandler()],
                              temperature=0.3)

        # Prompt Setup
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )

        # Memory Setup
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # LLMChain Initialization
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory
        )

    def chat(self):
        print("Hello! I'm your AI assistant created by LangChain + RAG. Let's have a conversation! \n Type 'quit' to end conversation")
        
        while True:
            user_input = input("You:")
            if user_input.lower() == "quit":
                break
            
            ai_output = self.conversation({"question": user_input})
            print(ai_output, end='')
        
        print("Goodbye!")

if __name__ == "__main__":
    chatbot = LangChainChatbot()
    chatbot.chat()
