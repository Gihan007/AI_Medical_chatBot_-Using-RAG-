
from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(
        data , 
        glob="*.pdf" , 
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents

documents = load_pdf_files(data = DATA_PATH)
print(len(documents))
print(documents[0].page_content)

def create_chunks(extracted_text):
    text_splitted = RecursiveCharacterTextSplitter(chunk_size = 500
                                                    , chunk_overlap = 100)
    text_chunks = text_splitted.split_documents(extracted_text)
    return text_chunks

import pandas as pd
text_chunks = create_chunks(documents)
#print(len(text_chunks))

li = []

for chunk in text_chunks:
    #print(chunk.page_content)
    k = chunk.page_content
    li.append(k)
    

print(li)

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embedding_model

embedding_model = get_embedding_model()

DB_FAISS_PATH = "vectorstoredb_faiss"
db = FAISS.from_documents(text_chunks , embedding_model)
db.save_local(DB_FAISS_PATH)

model = HuggingFaceHub(
    huggingfacehub_api_token="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",
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 256
    }
)


DB_FAISS_PATH = "vectorstoredb_faiss"

custom_prompt_template = """ 
use the peices of information provided in the context to answer user's question
If you dont know the answer , just say that u dont know , dont try to makeup any answer.
Dont provide anything which is out of the context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template , input_variables=["context" , "question"])
    return prompt

output_parser = StrOutputParser()
retriever = db.as_retriever(search_kwargs = {'k' :3})

prompt = set_custom_prompt(custom_prompt_template)

rag_chain = (
    {"context" : retriever , "question" : RunnablePassthrough()}
    | prompt
    | model
    |output_parser
)


user_query=input("Write Query Here: ")
response=rag_chain.invoke(user_query)
print("RESULT: ", response)



