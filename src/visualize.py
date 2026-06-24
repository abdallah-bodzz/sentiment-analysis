import os
import logging
from typing import List, Union
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
from .config import config

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')

def visualize_sentiment_distribution(
    sentiment_counts: Union[pd.Series, dict], 
    output_dir: str = None
):
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {
        'positive': '#2ecc71', 
        'negative': '#e74c3c', 
        'neutral': '#3498db'
    }
    
    if isinstance(sentiment_counts, dict):
        sentiment_counts = pd.Series(sentiment_counts)
    
    color_list = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=color_list)
    
    ax.set_title('Sentiment Distribution', fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Sentiment Class', fontsize=12)
    ax.set_ylabel('Number of Reviews', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., 
            height + 0.5,
            f'{int(height)}', 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    output_path = os.path.join(output_dir, 'sentiment_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()
    logger.info(f'Sentiment distribution saved to {output_path}')

def create_wordcloud(
    texts: List[str], 
    output_dir: str = None,
    max_words: int = None
):
    output_dir = output_dir or config.output_dir
    max_words = max_words or config.wordcloud_max_words
    os.makedirs(output_dir, exist_ok=True)
    
    all_text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=config.wordcloud_width,
        height=config.wordcloud_height,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
        prefer_horizontal=0.7,
        min_font_size=10
    ).generate(all_text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Reviews', fontsize=14, pad=20, fontweight='bold')
    
    output_path = os.path.join(output_dir, 'wordcloud.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()
    logger.info(f'Word cloud saved to {output_path}')

def visualize_confusion_matrix(conf_matrix: List[List[int]], output_dir: str = None):
    output_dir = output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['positive', 'negative', 'neutral'],
        yticklabels=['positive', 'negative', 'neutral']
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()
    logger.info(f'Confusion matrix saved to {output_path}')