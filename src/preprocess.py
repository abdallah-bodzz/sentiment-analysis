import re
import logging
from typing import List, Tuple, Optional
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

def _download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
            logger.info(f'Downloaded NLTK resource: {resource}')

_download_nltk_resources()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    
    return ' '.join(tokens)

def load_data(filepath: str, text_column: str = 'review') -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logger.info(f'Loaded {len(df)} records from {filepath}')
    except FileNotFoundError:
        logger.error(f'File not found: {filepath}')
        raise
    
    df['cleaned_review'] = df[text_column].apply(preprocess_text)
    logger.info(f'Preprocessed {len(df)} reviews')
    
    return df