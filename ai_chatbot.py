# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:55:39 2024

@author: Vivek Parihar
"""
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import openai
#from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from flask import Flask, render_template, request, redirect

from langchain.retrievers.merger_retriever import MergerRetriever
from text_to_speech import speak


OPENAI_API_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

#Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history, response

    if request.method == 'POST':
        user_question = request.form['user_question']
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']
        response = response['answer']

        

    return render_template('chat.html', chat_history=chat_history)
    #speak(response['answer'])
print('done----------------------')
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)