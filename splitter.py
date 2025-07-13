from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split input text into chunks using a recursive character splitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks
