from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from icecream import ic
import tiktoken
import numpy as np

load_dotenv()
# Define a custom User-Agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
# 
# Load Documents
loader = WebBaseLoader(
    web_paths=("https://sivaganesh07.github.io/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("content-wrapper", "header-content")
        )
    ),
    requests_kwargs={"headers": headers}  # Pass the headers here
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())



retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# docs = retriever.get_relevant_documents("What is his technical skills?")

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # Chain
# chain = prompt | llm

# # Run
# chain.invoke({"context":docs,"question":"What is his technical skills?"})


# prompt_hub_rag = hub.pull("rlm/rag-prompt")

# ic(prompt_hub_rag)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

ic(rag_chain.invoke("What projects he worked on?"))