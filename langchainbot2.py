# Load documents
from langchain.document_loaders import WebBaseLoader
url1 = "https://lilianweng.github.io/posts/2023-06-23-agent/"
url2 = "https://doi.org/10.48550/arXiv.2303.17564"
url3 = "https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/"
url4 = "https://raw.githubusercontent.com/handsomezebra/chatbot-example/main/backend_langchain/instruct_gpt.txt"
url5 = "https://arxiv.org/abs/2302.13971v1"
loader = WebBaseLoader([url1,url2,url3,url4,url5])

# Split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
splits = text_splitter.split_documents(loader.load())

# Embed and store splits
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Prompt 
# https://smith.langchain.com/hub/rlm/rag-prompt
from langchain import hub
rag_prompt = hub.pull("rlm/rag-prompt")

# LLM
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# RAG chain 
#from langchain.schema.runnable import RunnablePassthrough
#rag_chain = (
 #   {"context": retriever, "question": RunnablePassthrough()} 
 #   | rag_prompt 
 #   | llm 
#)
# Conversation chain
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

print("Creating chains")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

# Loop to keep the chatbot running
exit_conditions = (":q", "quit", "exit")
while True:
    query = input(">type ':q','quit' or 'exit' to end conversation ")
    if query in exit_conditions:
        break
    else:
        response = conversation(query)
        print(response["answer"])
