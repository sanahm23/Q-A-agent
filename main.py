# from langchain_community.document_loaders import PyPDFLoader

# from langchain_text_splitters import TokenTextSplitter

# from langchain_core.output_parsers.string import StrOutputParser
# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import (PromptTemplate, 
#                                     HumanMessagePromptTemplate, 
#                                     ChatPromptTemplate)
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import (ChatOpenAI, 
#                               OpenAIEmbeddings)
# from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv
import os
load_dotenv()
# # first task is to load it
# loader_pdf = PyPDFLoader("C:\\Users\\Sanah\\Desktop\\testing-langchain.pdf")

# docs_list = loader_pdf.load()

# # Text splitters break large docs into smaller chunks that will be retrievable individually and fit within model context window limit
# token_splitter = TokenTextSplitter(encoding_name="cl100k_base",
#                                    chunk_size=800,
#                                    chunk_overlap=100)
# docs_list_tokens_split = token_splitter.split_documents(docs_list)

# # embeddings
# embedding = OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.getenv("OPENAI_API_KEY")
# )

# # vector db stores data
# vectorstore = Chroma.from_documents(documents = docs_list_tokens_split, embedding = embedding, persist_directory= "./intro_to_ai")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

loader_pdf = PyPDFLoader(
    r"C:\Users\Sanah\Desktop\testing-langchain.pdf"
)
docs = loader_pdf.load()

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
    persist_directory="./intro_to_ai"
)
