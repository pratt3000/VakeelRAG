from langchain.docstore.document import Document
import pandas as pd

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

def create_vectordb():
    df = pd.read_csv("Constitution_of_India.csv")
    chunks = []

    for _, row in df.iterrows():
        ref, content= row['article'], row['title'] + row['description']
        chunks.append(Document(page_content=content, metadata={"source": ref}))

    client = weaviate.Client(
        embedded_options = EmbeddedOptions()
    )

    vectorstore = Weaviate.from_documents(
        client = client,    
        documents = chunks,
        embedding = OpenAIEmbeddings(),
        by_text = False
    )

    retriever = vectorstore.as_retriever()
    return retriever
