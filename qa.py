from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

def get_qa_chain(vector_store, k=3):
    """
    Create a RetrievalQA chain with a chat LLM and FAISS retriever.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    logger.info(f"Retriever created (top {k} docs)")

    # Explicitly provide the API key from env
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    logger.info("RetrievalQA chain created")
    return qa_chain

def answer_question(qa_chain, question):
    """
    Run the QA chain on a question. Returns the answer text.
    """
    logger.info(f"Running QA chain for question: {question}")
    try:
        result = qa_chain.run(question)
    except Exception as e:
        logger.error(f"Error during QA inference: {e}")
        raise e
    return result
