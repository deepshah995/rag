import pdfminer
from pdfminer.high_level import extract_pages, extract_text
import streamlit as st
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI



import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-4o-mini")

st.title("PDF Question Answering App")

st.write(pdfminer.__version__)  

uploaded_file = st.file_uploader("Choose a file", "pdf")


if uploaded_file is not None:
    text = extract_text(uploaded_file)

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Convert the extracted text into a Document object
    docs = [Document(page_content=text)]  

    # Split the document into smaller chunks
    splits = text_splitter.split_documents(docs) 
    import chromadb

    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")


    def format_docs(text):
        return text


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.write("Summary of the doc:")
    st.write(rag_chain.invoke("Give me a brief of the text"))

    prompt = st.text_input("Enter your prompt")
    if prompt:
        response = rag_chain.invoke(prompt)
        st.write(response)

else:
    st.info("Please upload a PDF file to extract text.")
