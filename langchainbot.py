from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
from langchain.schema import SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory

# LLM
#llm = ChatOpenAI()
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature=0.3)

# Prompt 
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)



#llm = OpenAI(temperature=0.7)
#memory = ConversationBufferMemory()
#chain = ConversationChain(llm=llm, memory=memory)

print("Hello! I'm your AI assistant created by LangChain + RAG. Let's have a conversation!")

while True:
    user_input = input("You:")
    if user_input.lower() == "quit":
        break
    
    ai_output = conversation({"question": user_input})
    print(ai_output, end='')

print("Goodbye!")

# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
#conversation({"question": "hi"})
#conversation({"question": "Translate this sentence from English to French: I love programming."})



#chat([HumanMessage(content="Translate this sentence to French: I love programming.")])


#resp = chat([HumanMessage(content="Write me a song about sparkling water.")])









