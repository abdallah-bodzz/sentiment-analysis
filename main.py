import os
import sys
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.preprocess import load_data
from src.sentiment import analyze_sentiments, evaluate_model
from src.visualize import (
    visualize_sentiment_distribution, 
    create_wordcloud,
    visualize_confusion_matrix
)

def setup_logging():
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(f'{config.output_dir}/pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info('Starting Sentiment Analysis Pipeline')
    
    try:
        df = load_data(config.data_path, config.text_column)
        logger.info(f'Data loaded: {len(df)} records')
        
        df, sentiment_counts = analyze_sentiments(df)
        logger.info(f'Analysis complete: {sentiment_counts.to_dict()}')
        
        eval_results = evaluate_model(df)
        
        visualize_sentiment_distribution(sentiment_counts)
        create_wordcloud(df[config.text_column].tolist())
        
        if eval_results and 'confusion_matrix' in eval_results:
            visualize_confusion_matrix(eval_results['confusion_matrix'])
        
        output_path = f'{config.output_dir}/analyzed_reviews.csv'
        df.to_csv(output_path, index=False)
        logger.info(f'Results saved to {output_path}')
        
        print('\n' + '='*60)
        print('SENTIMENT ANALYSIS RESULTS')
        print('='*60)
        print(f'\nTotal Reviews: {len(df)}')
        print(f'Accuracy: {eval_results["accuracy"]:.2%}' if eval_results else '')
        print(f'\nSentiment Distribution:')
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(df)) * 100
            print(f'  {sentiment.capitalize()}: {count} ({pct:.1f}%)')
        print('\nSample Results:')
        print(df[[config.text_column, 'sentiment']].head(5).to_string(index=False))
        print('\n' + '='*60)
        
        logger.info('Pipeline completed successfully')
        
    except Exception as e:
        logger.error(f'Pipeline failed: {e}', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()