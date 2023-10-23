from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LangChainChatbot:

    def __init__(self):
        print("Loading documents")
        loader = DirectoryLoader('./', glob="*.txt")
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(documents)

        print("Creating embeddings")
        # Create embeddings for each chunk
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        print("Creating chains")
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

        self.chain = ConversationChain(llm=llm, memory=memory, verbose=True)

    def chat(self):
        print("Hello! I'm your AI assistant created by LangChain. Let's have a conversation!")
        while True:
            user_input = input("> ")
            if user_input.lower() == "quit":
                break
            ai_output = self.chain(user_input)
            print(ai_output, end='')
        print("Goodbye!")

if __name__ == "__main__":
    chatbot = LangChainChatbot()
    chatbot.chat()
