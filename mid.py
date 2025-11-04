from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from getEmbeddings import getEmbeddings


pdf_path = "PDFS/About Me.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

if not docs:
    raise ValueError("I Can't find any text in PDF")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)


embeddings = getEmbeddings()
vectorStore = FAISS.from_documents(chunks, embeddings)
similar = vectorStore.as_retriever()


llm = Ollama(model="llama3.1:8b")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the information from the document.

    Question: {question}
    Context: {context}
    """
)

chain = (
    RunnableMap({
        "context": similar,
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
)


print("Type your question in English\n")

while True:
    query = input("Question: ")
    if query in ["exit", "quit"]:
        break

    answer = chain.invoke({"question": query})
    print("\nAnswer:")
    print(answer)

