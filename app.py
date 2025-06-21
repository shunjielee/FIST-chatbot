import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_vector_store(docs_folder="docs"):
    """
    Processes all PDF files in the specified folder, splits them into chunks,
    creates embeddings, and stores them in a FAISS vector store.
    Returns the vector store.
    """
    if not os.path.exists(docs_folder):
        st.error(f"The '{docs_folder}' directory does not exist. Please create it and add your PDF files.")
        return None

    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in the '{docs_folder}' directory.")
        return None

    docs = []
    with st.spinner("Processing PDF documents..."):
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(docs_folder, pdf_file))
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        # Using a locally-runnable, open-source embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectors = FAISS.from_documents(final_documents, embeddings)
    
    st.success("Vector store created successfully!")
    return vectors

def main():
    st.set_page_config(page_title="FIST Industrial Training chatbot", layout="wide")
    
    st.title("FIST Industrial Training chatbot")
    st.markdown("""
        Welcome! This app allows you to ask questions about the FIST Industrial Training Programme.
        
        **How it works:**
        1. The PDF documents in the `docs` folder have been processed.
        2. Ask any question about the documents in the chat box below.
        3. The chatbot will use the document content to provide answers.
    """)

    # Use Streamlit's secrets management for the API key. This is for deployment.
    # For local testing, you would create a file .streamlit/secrets.toml and add your key there.
    # Example .streamlit/secrets.toml:
    # GROQ_API_KEY = "YOUR_API_KEY_HERE"
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = groq_api_key
    except KeyError:
        st.error("Groq API Key not found! Please add it to your Streamlit secrets.")
        st.info("When deploying on Streamlit Cloud, you must set the GROQ_API_KEY in the app's secrets management.")
        st.stop()
    
    with st.sidebar:
        st.success("API Key loaded successfully!")
        st.info("The chatbot is ready and is using the documents provided in the `docs` folder.")

    # Session state initialization
    if "vectors" not in st.session_state:
        st.session_state.vectors = create_vector_store()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    vectors = st.session_state.vectors

    if vectors is None:
        st.warning("Vector store is not available. Please check the 'docs' folder and PDF files.")
        st.stop()

    # Initialize the language model
    llm = ChatGroq(model_name="Llama3-70b-8192", temperature=0.7)

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the user's questions based on the provided context.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a detailed answer based on the context.
        <context>
        {context}
        </context>
        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User input
    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history 
            })
            
            answer = response.get("answer", "I could not find an answer.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main() 