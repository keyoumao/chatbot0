from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import SVMRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder, 
                               SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.memory import (ConversationSummaryMemory, ConversationSummaryBufferMemory, 
                             ConversationBufferMemory)
# Prompt 
# https://smith.langchain.com/hub/rlm/rag-prompt
from langchain import hub
#from langchain.schema import ContextSchema, QuestionSchema




class LangChainChatbot:

    def __init__(self):
        # Load documents
        self.loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        self.data = self.loader.load()
        
        # Split documents
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.splits = self.text_splitter.split_documents(self.loader.load())
        self.all_splits = self.text_splitter.split_documents(self.data)

        # Embed and store splits
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(documents=self.splits, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        # Prompt
        #self.context_schema = ContextSchema(text="{context}")
        #self.question_schema = QuestionSchema()
  
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

        # LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        
        # RAG chain
        self.chain = ConversationalRetrievalChain(retriever=self.retriever, prompt=self.prompt, llm=self.llm, verbose=True)
        # this one has a problem of saving chat history!



    def chat(self):
        print("Hello! I'm your AI assistant created by LangChain. Let's have a conversation!\\n Type 'quit' to end conversation")
        #self.context_schema = ContextSchema(text="{context}")
        #self.question_schema = QuestionSchema()
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break
            # Update question schema with user input
            #self.question_schema.update(text=user_input)
               # Construct prompt each loop
            #self.prompt = ChatPromptTemplate(messages=[self.context_schema,self.question_schema])
               
            ai_output = self.chain(user_input)
            #ai_output=self.chain(prompt=prompt, context={"context": ""}, question=user_input)
            
            print(ai_output["answer"], end='')

        print("Goodbye!")
        
if __name__ == "__main__":
    chatbot = LangChainChatbot()
    chatbot.chat()


