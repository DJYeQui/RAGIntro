import bs4
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

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load, chunk and index the contents of the blog.
# for web scrapping RAG datas
"""loader = WebBaseLoader(
    web_paths=("https://people.ieu.edu.tr/tr/senemkumovametin",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer("div", id = "short_cv")
    ),
)
docs = loader.load()
print(docs)"""
# for txt read
with open('RAGInfo.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    print(content)
    documentTXT = [Document(page_content=content),
                   Document(page_content="Fatih Anamasli Senem Hocanın asistanı olarak görev yapmaktadır."),
                   Document(
                       page_content="İzmir ekonomi öğrencisidir. Başarılı bir yazılımcı  ve game designerdır. Geçmişinde girişimcilik maceraları olmuştur ve şirketlerde ve vc lerde çalışmıltır."),
                   Document(
                       page_content="I am someone who always tries to improve social skills and software knowledge. "
                                    "I have made it my goal to experience many different fields and expand my perspective with the information I have gained from those fields. "
                                    "I believe that experience in each different department will provide different abilities.")]
    print(documentTXT)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(documentTXT)
vectorstore = Chroma.from_documents(documents=documentTXT, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

if __name__ == "__main__":
    while (True):
        do_continue = input("do you have more questions? (y/n)")
        if do_continue == "y":
            enter = input("what is your question")
            for chunk in rag_chain.stream(enter):
                print(chunk, end="", flush=True)
        else:
            break
