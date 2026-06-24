import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_text
from src.sentiment import get_sentiment

class TestSentimentAnalysis(unittest.TestCase):
    
    def test_preprocess_text(self):
        text = "This is a test sentence!"
        result = preprocess_text(text)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "test sentence")
    
    def test_get_sentiment(self):
        self.assertEqual(get_sentiment("I love this"), 'positive')
        self.assertEqual(get_sentiment("I hate this"), 'negative')
        self.assertEqual(get_sentiment("It's ok"), 'neutral')
    
    def test_empty_input(self):
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(get_sentiment(""), 'neutral')

if __name__ == '__main__':
    unittest.main()