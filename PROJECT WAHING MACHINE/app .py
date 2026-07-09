import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot for Technical Documentation")

# Read API key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

if "OPENAI_API_KEY" not in os.environ:
    st.warning("Please enter your OpenAI API Key.")
    st.stop()

HTML_FILE = "1.html"

if not os.path.exists(HTML_FILE):
    st.error(f"{HTML_FILE} not found.")
    st.stop()

with st.spinner("Loading document..."):

    loader = UnstructuredHTMLLoader(HTML_FILE)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

question = st.text_input("Ask a question")

if st.button("Get Answer"):

    if question:

        with st.spinner("Generating answer..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            messages = prompt.format_messages(
                context=context,
                question=question
            )

            response = llm.invoke(messages)

            st.success(response.content)
