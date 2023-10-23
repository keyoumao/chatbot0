from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

class Chatbot:
    def __init__(self, urls=None):
        if urls is None:
            urls = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://medium.com/codex/bloomberggpt-the-first-large-language-model-for-finance-61cc92075075",
                "https://raw.githubusercontent.com/handsomezebra/chatbot-example/main/backend_langchain/instruct_gpt.txt",
                "https://onlinelibrary.wiley.com/doi/full/10.1111/1911-3846.12832?casa_token=-WaGu4knKf8AAAAA%3Akxm2cS_VYEKHcg1j4yhkvq8VDoV5s6IWG_82sUvU1i7XEriwN5n3_kARJWWXRF3hL5jxI9vph7q0ynMO"
            ]

        # Load documents
        loader = WebBaseLoader(urls)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        splits = text_splitter.split_documents(loader.load())

        # Embed and store splits
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()

        # Prompt 
        rag_prompt = hub.pull("rlm/rag-prompt")

        # LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Conversation chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

    def run(self):
        # Loop to keep the chatbot running
        exit_conditions = (":q", "quit", "exit")
        while True:
            query = input(">type ':q','quit' or 'exit' to end conversation ")
            if query in exit_conditions:
                break
            else:
                response = self.conversation(query)
                print(response["answer"])

if __name__ == "__main__":
    bot = Chatbot()
    bot.run()
