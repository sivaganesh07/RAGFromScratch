
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

retriever = vectorstore.as_retriever()

# Find one nearby neighbours between nodes
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents("What is his experience?")
ic(len(docs))

#### RETRIEVAL and GENERATION ####

# # Prompt
# prompt = hub.pull("rlm/rag-prompt")

# # LLM
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

# # Post-processing
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Question
# response = rag_chain.invoke("What is the experience?")

# ic(response)



# Documents
# question = "What kinds of pets do I like?"
# document = "My favorite pet is a cat."


# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# # ic(num_tokens_from_string(question, "cl100k_base"))


# embd = OpenAIEmbeddings()
# query_result = embd.embed_query(question)
# document_result = embd.embed_query(document)
# # ic(len(query_result))
# # ic(len(document_result))

# def cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)

# similarity = cosine_similarity(query_result, document_result)
# ic("Cosine Similarity:", similarity)



