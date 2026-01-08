import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PDF_PATH = "data/Intro_to_AI_ques.pdf"
PERSIST_DIR = "./intro_to_ai"

@st.cache_resource
def load_retriever():
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1, "lambda_mult": 0.7}
    )

retriever = load_retriever()

system_prompt = SystemMessage(
    "Answer the question using only the provided content."
)

human_prompt = HumanMessagePromptTemplate.from_template(
    "Question:\n{question}\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate([system_prompt, human_prompt])

chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

st.header("365 Q&A Chatbot", divider=True)

question = st.text_input("Type your question")

if st.button("Ask") and question:
    placeholder = st.empty()
    answer = ""

    for chunk in chain.stream(question):
        answer += chunk
        placeholder.markdown(answer)
