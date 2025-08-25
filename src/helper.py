from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
#creating function to load my pdf and extract text from it
def load_pdf_file(data):
  loader=DirectoryLoader(
    data,
    glob="*.pdf",
    loader_cls=PyPDFLoader
  )

  document=loader.load()
  return document

def filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
  minimal_docs:List[Document]=[]

  for doc in docs:
    src=doc.metadata.get("source")
    minimal_docs.append(
      Document(
        page_content=doc.page_content,
        metadata={"source":src}
      )
    )
  return minimal_docs

# #split document into smaller chunks

def text_split(minimal_docs):
  text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20

  )
  texts_chunk= text_splitter.split_documents(minimal_docs)
  return texts_chunk

#download embedding model
def download_embeddings():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings
