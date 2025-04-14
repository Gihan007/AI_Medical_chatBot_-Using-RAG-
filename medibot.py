import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

# === Setup Vectorstore and Retriever ===
DB_FAISS_PATH = "vectorstoredb_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# === Custom Prompt ===
custom_prompt_template = """ 
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up any answer.
Don't provide anything which is out of the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# === Load Hugging Face Model ===


model = HuggingFaceHub(
    huggingfacehub_api_token="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" , 
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 256
    }
)


# === Build RAG Chain ===
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# === Streamlit UI ===
st.set_page_config(page_title="📄 MediBot", layout="centered")
st.title("🩺 AI Medical PDF ChatBot")
st.write("Ask a question based on the uploaded PDFs.")

# Initialize session state to hold chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("💬 Ask your question:")

if user_input:
    with st.spinner("Searching..."):
        response = rag_chain.invoke(user_input)
        # Extract clean answer
        clean_answer = response.split("Answer:")[-1].strip()

        # Add to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("MediBot", clean_answer))

# Display chat history
# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑‍💬 {speaker}:** {message}")
    else:
        st.markdown(f"**🤖 {speaker}:** {message}")
