
---

# 🩺 MediBot - AI Medical PDF ChatBot

MediBot is an AI-powered chatbot designed to assist users in querying medical PDFs using advanced NLP and retrieval techniques. It leverages state-of-the-art models like **Mistral-7B** and **sentence-transformers** to create an efficient Retrieval-Augmented Generation (RAG) system.

---

## 🚀 Features

- Chat with multiple medical PDF documents.
- Built with **LangChain**, **FAISS**, and **Streamlit**.
- Uses **sentence-transformers/all-mpnet-base-v2** for embedding.
- Powered by **Mistral-7B-Instruct** for intelligent responses.
- Context-aware responses strictly based on document content.

---

## 🧠 How It Works

### Step 1: Load PDF Files
```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
```

Each document is loaded and paginated as individual `page_content` elements.

---

### Step 2: Split Documents into Chunks
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
text_chunks = text_splitter.split_documents(documents)
```

Chunks allow more accurate vector search and better context extraction.

---

### Step 3: Generate Embeddings and Create Vector Store
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local("vectorstoredb_faiss")
```

We use FAISS to store vector representations of all text chunks for fast semantic retrieval.

---

### Step 4: Setup Mistral LLM via HuggingFace
```python
from langchain.llms import HuggingFaceHub

model = HuggingFaceHub(
    huggingfacehub_api_token="YOUR_TOKEN",
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)
```

---

### Step 5: Custom Prompt and RAG Chain
```python
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up any answer.
Don't provide anything which is out of the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
retriever = db.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

---

### Step 6: Build the Streamlit App
```python
import streamlit as st

st.set_page_config(page_title="📄 MediBot", layout="centered")
st.title("🩺 AI Medical PDF ChatBot")

# Chat UI and history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("💬 Ask your question:")
if user_input:
    with st.spinner("Searching..."):
        response = rag_chain.invoke(user_input)
        clean_answer = response.split("Answer:")[-1].strip()
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("MediBot", clean_answer))

# Display chat history
for speaker, message in st.session_state.chat_history:
    role = "🧑‍💬" if speaker == "You" else "🤖"
    st.markdown(f"**{role} {speaker}:** {message}")
```

---

## ✅ Requirements

- Python 3.10+
- LangChain
- HuggingFace Hub
- FAISS
- Streamlit
- sentence-transformers

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```
├── app.py                  # Streamlit frontend
├── data/                   # Your PDF files
├── vectorstoredb_faiss/    # FAISS vector DB
├── utils/                  # Optional helper functions
└── README.md
```

---

## 💡 Future Improvements

- File upload feature in Streamlit
- User authentication and storage
- Advanced summarization and visualization of data

---

## 🤝 Contributing

Contributions are welcome! Open a PR or issue to start improving the project.

---

## 📜 License

This project is licensed under the MIT License.

---

