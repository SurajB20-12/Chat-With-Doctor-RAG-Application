from math import e
from src.helper import load_pdf_file,filter_to_minimal_docs,text_split,download_embeddings

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY") 

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

extract_data=load_pdf_file(data='data/')
filter_data=filter_to_minimal_docs(extract_data)
text_chunk=text_split(filter_data)

embeddings=download_embeddings()

pinecone_api_key=PINECONE_API_KEY

pc=Pinecone(api_key=pinecone_api_key)

index_name="medical-chatbot"

if not pc.has_index(index_name):
  pc.create_index(
    name=index_name,
    dimension=384, # Set the dimension to match your embeddings
    metric="cosine",# Set the metric to use for similarity search
    spec=ServerlessSpec(
      cloud="aws",
      region="us-east-1"

    )

  )
index=pc.Index(index_name) 

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    index_name=index_name
)