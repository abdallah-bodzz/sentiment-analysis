import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    data_path: str = 'data/sample_reviews.csv'
    output_dir: str = 'outputs'
    text_column: str = 'review'
    label_column: str = 'label'
    sentiment_threshold: float = 0.2
    log_level: str = 'INFO'
    random_seed: int = 42
    wordcloud_max_words: int = 100
    wordcloud_width: int = 800
    wordcloud_height: int = 400
    figure_dpi: int = 300
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

config = Config()