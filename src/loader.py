import logging
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

def load_pdf(file_list):
    """
    Read a list of uploaded PDF files and extract all text.
    """
    text = ""
    for pdf_file in file_list:
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info(f"Extracted text from {len(reader.pages)} pages of {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error reading {getattr(pdf_file, 'name', pdf_file)}: {e}")
            raise e
    return text
