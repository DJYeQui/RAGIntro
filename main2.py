import chainlit as cl
from bs4 import SoupStrainer
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Loading content from a text file
with open('RAGInfo.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    documentTXT = [
        Document(page_content=content),
        Document(page_content="Fatih Anamasli Senem Hocanın asistanı olarak görev yapmaktadır."),
        Document(page_content="İzmir ekonomi öğrencisidir. Başarılı bir yazılımcı ve game designerdır. "
                              "Geçmişinde girişimcilik maceraları olmuştur ve şirketlerde ve vc lerde çalışmıltır."),
        Document(page_content="I am someone who always tries to improve social skills and software knowledge. "
                              "I have made it my goal to experience many different fields and expand my perspective with the information I have gained from those fields. "
                              "I believe that experience in each different department will provide different abilities.")
    ]

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(documentTXT)

# Create a Chroma vectorstore from the documents
vectorstore = Chroma.from_documents(documents=documentTXT, embedding=OpenAIEmbeddings())

# Create a retriever
retriever = vectorstore.as_retriever()

# Load the RAG prompt template from the hub
prompt = hub.pull("rlm/rag-prompt")


# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Create a RAG chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


# Chainlit application setup
@cl.on_message
async def handle_message(message: cl.message.Message):
    # Extract the text content from the Chainlit message object
    user_input = message.content
    # Log the input message for debugging
    print(f"Received message: {user_input}")

    # Try to get the response from the RAG chain
    try:
        response = ""
        # Get the result from the chain (without async for, use regular for)
        result = rag_chain.invoke(user_input)

        # Iterate over the chunks and build the response
        for chunk in result:
            response += chunk
            print(f"Chunk received: {chunk}")  # Debug log

        # Send the response back to the user
        if response:
            await cl.Message(content=response).send()
        else:
            await cl.Message(content="No relevant information found.").send()

    except Exception as e:
        print(f"Error: {e}")
        await cl.Message(content="An error occurred while processing your request.").send()
