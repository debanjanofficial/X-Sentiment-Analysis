import re
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextProcessor:
    
    def __init__(self, positive_words=None, negative_words=None):
        self.positive_words = positive_words if positive_words else set()
        self.negative_words = negative_words if negative_words else set()
    
    def preprocess_text(self, text):
        """Clean and preprocess tweet text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_lexicon_features(self, text):
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        
        # Calculate sentiment score
        lexicon_score = (positive_count - negative_count) / max(len(tokens), 1)
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'lexicon_score': lexicon_score
        }
