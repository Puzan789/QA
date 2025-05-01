import logging
import re 
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # Prevent adding multiple handlers
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler('botanza.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger

def clean_text(text):
    """
    Clean the text by removing special characters and converting to lowercase.
    """
    text=text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text
