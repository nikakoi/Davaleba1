from langchain_community.embeddings.ollama import OllamaEmbeddings


def getEmbeddings():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings