import os
import time
from unittest import loader
from wsgiref import headers
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

placeholder = st.empty()
progress_bar = st.progress(0)

# === URLs to Load ===

from urls import urls_dict
urls = list(urls_dict.values())
    
    
# === Document Loading and Text Splitting ===

@st.cache_resource
def load_documents(urls=urls):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    loader = UnstructuredURLLoader(urls=urls, headers=headers)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    return splits


# === Embeddings and Vector Store ===

def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("EvoAtlas-faiss_index")
    return vectorstore

@st.cache_resource
def read_vectorstore(splits):
    if os.path.exists("EvoAtlas-faiss_index"):
        vectorstore = FAISS.load_local("EvoAtlas-faiss_index", OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")), allow_dangerous_deserialization=True)
    else:
        vectorstore = create_vectorstore(splits)
    return vectorstore


# === Create LLM and Prompt Template ===

@st.cache_resource
def create_llm_and_prompt():
    llm = OpenAI(temperature=0.7, max_tokens=1024, openai_api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Answer in the language of the question. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    return llm, prompt


# === Create Stuff and Retrieval Chain ===

@st.cache_resource
def create_chains(llm, prompt, _vectorstore):
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), qa_chain)
    return rag_chain


# === Template UI Design ===    

st.title("EvoAtlas")
st.sidebar.title("EvoAtlas Topics")
for i, (topic, url) in enumerate(urls_dict.items()):
    st.sidebar.write(f"{i+1}. [{topic}]({url})")
    
placeholder.info("Loading and processing the documents. This may take a while...")
splits = load_documents()
time.sleep(2)
progress_bar.progress(30)

placeholder.info("Creating embeddings and vector store.")
progress_bar.progress(60)
vectorstore = read_vectorstore(splits)

placeholder.info("Setting up the language model and retrieval chain.")
llm, prompt = create_llm_and_prompt()
rag_chain = create_chains(llm, prompt, vectorstore)

placeholder.success("EvoAtlas is ready to use!")
progress_bar.progress(100)
time.sleep(3)
placeholder.empty()
progress_bar.empty()

    
query = st.text_input("Enter your question about Evolutionary Biology:", key="user_question",
                      placeholder="e.g., How does genetic drift work?")
if query:
    if os.path.exists("EvoAtlas-faiss_index"):
        response = rag_chain.invoke({"input": query}, return_only_outputs=True)
        source_urls = set([context.metadata["source"] for context in response["context"]])
        st.write(response["answer"])
        st.write("<br/>", unsafe_allow_html=True)
        st.write(f"**Sources:**")
        for url in source_urls:
            st.write(f"* {url}")