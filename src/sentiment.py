import logging
from typing import Tuple, Optional, Dict
import pandas as pd
from textblob import TextBlob
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
from .config import config

logger = logging.getLogger(__name__)

# Custom neutral words to override TextBlob
NEUTRAL_WORDS = {
    'okay', 'ok', 'average', 'mediocre', 'neutral', 'nothing special', 
    'fine', 'decent', 'acceptable', 'alright', 'so-so', 'ordinary'
}

def get_sentiment(text: str) -> str:
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
    except Exception as e:
        logger.warning(f'TextBlob error on: {text[:50]}... - {e}')
        return 'neutral'
    
    # Check for neutral indicators
    text_lower = text.lower()
    for word in NEUTRAL_WORDS:
        if word in text_lower:
            return 'neutral'
    
    if polarity > config.sentiment_threshold:
        return 'positive'
    elif polarity < -config.sentiment_threshold:
        return 'negative'
    return 'neutral'

def analyze_sentiments(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df['sentiment'] = df['cleaned_review'].apply(get_sentiment)
    sentiment_counts = df['sentiment'].value_counts()
    
    logger.info(f'Sentiment distribution: {sentiment_counts.to_dict()}')
    return df, sentiment_counts

def evaluate_model(df: pd.DataFrame) -> Optional[Dict]:
    if 'label' not in df.columns:
        logger.warning('No labels found for evaluation')
        return None
    
    y_true = df['label'].str.lower()
    y_pred = df['sentiment']
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    
    logger.info(f'Model Accuracy: {accuracy:.2%}')
    
    report_text = classification_report(y_true, y_pred, zero_division=0)
    logger.info(f'\n{report_text}')
    
    report_path = f'{config.output_dir}/classification_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'sentiment_distribution': df['sentiment'].value_counts().to_dict()
        }, f, indent=2)
    
    logger.info(f'Evaluation results saved to {report_path}')
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix
    }