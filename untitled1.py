# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:09:03 2024

@author: Admin
"""
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import openai
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstore import FAISS


from langchain_community.embeddings.openai import OpenAIEmbeddings
#from langchain_community.vectorstores import FAISS

def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Example usage:
file_path = r'C:\Users\Admin\OneDrive - bizmetric.com\Documents\edtech\pdfGPT-main\PDFs\EDA_Zep.pdf'  # Replace 'example.pdf' with the path to your PDF file
text = read_pdf(file_path)
print(text)


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

chunks= get_text_chunks(text)
print(chunks)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore




'''def get_vectorstore(text_chunks):
    #openai = ChatOpenAI(model_name="gpt-3.5-turbo")
    embeddings = openai.api_resources.embedding.Embedding(model_name="gpt-3.5-turbo", texts=text_chunks)
    vectorstore = FAISS.from_embeddings(text_chunks, embeddings)
    return vectorstore'''


vector= get_vectorstore(chunks)
print(vector)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

conversation= get_conversation_chain(vector)
print(conversation)


response = conversation({'question': 'What is Data Analysis ?'})
print(response)