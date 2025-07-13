import streamlit as st
from dotenv import load_dotenv
import os
import logging

# Load environment and configure logging
load_dotenv()
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from loader import load_pdf
from splitter import split_text
from vectorstore import create_vector_store, save_vector_store, load_vector_store
from qa import get_qa_chain, answer_question
from langchain_openai import ChatOpenAI  # ensure langchain-openai is installed

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")
    st.title("📖 LangChain PDF Chatbot")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔥 Missing OPENAI_API_KEY in .env. Please set it and restart.")
        return

    # Sidebar: PDF upload and processing
    with st.sidebar:
        st.header("Upload & Process PDF")
        pdf_files = st.file_uploader(
            "Select PDF file(s)", type=['pdf'], accept_multiple_files=True
        )
        if st.button("Process PDFs"):
            if not pdf_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Reading and processing PDFs..."):
                    try:
                        # 1. Load PDF text
                        raw_text = load_pdf(pdf_files)
                        # 2. Split text into chunks
                        chunks = split_text(raw_text)
                        # 3. Create and save FAISS vector store
                        vector_db = create_vector_store(chunks)
                        save_vector_store(vector_db, directory="faiss_index")
                        st.success("✅ PDFs processed successfully.")
                        st.write(f"Indexed {len(chunks)} text chunks.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")

    # Main area: QA interface
    st.header("Ask a Question")
    query = st.text_input("Enter your question about the PDF content:")
    if query:
        try:
            # Load (or recreate) the FAISS vector store
            if os.path.exists("faiss_index"):
                vector_db = load_vector_store(directory="faiss_index")
                qa_chain = get_qa_chain(vector_db)
                answer = answer_question(qa_chain, query)
                st.markdown(f"**Answer:** {answer}")
            else:
                st.error("⚠️ No processed PDFs found. Upload and process PDFs first.")
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            logger.exception("QA chain failed")

if __name__ == "__main__":
    main()
