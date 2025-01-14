import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retrieval_qa' not in st.session_state:
    st.session_state.retrieval_qa = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def create_vectordb(pdf_file):
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Process the PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
    vector_store = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
    )
    
    # Remove temporary file
    os.remove("temp.pdf")
    return vector_store

def setup_qa_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model="gpt-4-turbo-preview")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

# Streamlit UI
st.title("PDF Question Answering System")

# File upload
uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])

if uploaded_file and st.session_state.vector_store is None:
    with st.spinner('Processing PDF...'):
        st.session_state.vector_store = create_vectordb(uploaded_file)
        st.session_state.retrieval_qa = setup_qa_chain(st.session_state.vector_store)
    st.success('PDF processed successfully!')

# Chat interface
if st.session_state.vector_store is not None:
    # Display chat history
    for q, a, sources in st.session_state.chat_history:
        st.write("Question:", q)
        st.write("Answer:", a)
        
        # Display historical sources in an expander
        with st.expander("📚 View Sources", expanded=False):
            st.markdown("---")
            for i, doc in enumerate(sources, 1):
                with st.container():
                    st.markdown(f"**Source {i}** (Page {doc.metadata.get('page', 'Unknown')})")
                    st.markdown("""
                    ```text
                    {}
                    ```
                    """.format(doc.page_content))
            st.markdown("---")

    # Question input
    question = st.text_input("Ask a question about your PDF:")
    if st.button("Ask"):
        if question:
            with st.spinner('Finding answer...'):
                result = st.session_state.retrieval_qa({"query": question})
                answer = result['result']
                source_documents = result['source_documents']
                
                # Store in chat history
                st.session_state.chat_history.append((question, answer, source_documents))
                
                # Display the latest Q&A with sources in a cleaner format
                st.write("Question:", question)
                st.write("Answer:", answer)
                
                # Display sources in an expander
                with st.expander("📚 View Sources", expanded=False):
                    st.markdown("---")
                    for i, doc in enumerate(source_documents, 1):
                        with st.container():
                            st.markdown(f"**Source {i}** (Page {doc.metadata.get('page', 'Unknown')})")
                            st.markdown("""
                            ```text
                            {}
                            ```
                            """.format(doc.page_content))
                    st.markdown("---")

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
