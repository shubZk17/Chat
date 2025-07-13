import logging
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

logger = logging.getLogger(__name__)

def create_vector_store(text_chunks):
    """
    Embed text chunks and create a FAISS vector store.
    """
    logger.info(f"Creating embeddings for {len(text_chunks)} chunks")
    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def save_vector_store(vector_store, directory="faiss_index"):
    """
    Save FAISS index to disk for persistence.
    """
    vector_store.save_local(directory)
    logger.info(f"Vector store saved to {directory}/")

def load_vector_store(directory="faiss_index"):
    """
    Load a saved FAISS index from disk. Returns a FAISS vector store.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Vector store loaded from {directory}/")
    return vector_store
