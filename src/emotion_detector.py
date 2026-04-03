"""
Emotion Detection from Text
Main EmotionDetector class for predicting emotions from text
"""

import re
import joblib
import numpy as np
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class EmotionDetector:
    """
    A class for detecting emotions from text using machine learning.
    """
    
    # Emotion labels
    EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the EmotionDetector.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file
        """
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('wordnet')
            
        self.stop_words = set(stopwords.words('english'))
        
        if model_path:
            self.load_model(model_path)
        else:
            self._create_default_pipeline()
    
    def _create_default_pipeline(self):
        """Create a default ML pipeline with TF-IDF and Logistic Regression."""
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ])
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text for emotion detection.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict(self, text: str) -> str:
        """
        Predict emotion from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Predicted emotion
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict([processed_text])[0]
        
        return self.EMOTIONS[prediction] if isinstance(prediction, int) else prediction
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get prediction probabilities for all emotions.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary with emotion probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        processed_text = self.preprocess_text(text)
        probabilities = self.model.predict_proba([processed_text])[0]
        
        return dict(zip(self.EMOTIONS, probabilities))
    
    def train(self, texts: List[str], emotions: List[str]):
        """
        Train the emotion detection model.
        
        Args:
            texts (List[str]): List of text samples
            emotions (List[str]): List of corresponding emotion labels
        """
        if len(texts) != len(emotions):
            raise ValueError("Number of texts and emotions must be equal.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Train the model
        self.model.fit(processed_texts, emotions)
        
        print("Model training completed!")
    
    def save_model(self, path: str):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a pre-trained model from disk.
        
        Args:
            path (str): Path to the model file
        """
        try:
            self.model = joblib.load(path)
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {path}")
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List]:
        """
        Get the most important features for each emotion class.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Dictionary with top features for each emotion
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Get feature names and coefficients
        feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
        coefficients = self.model.named_steps['classifier'].coef_
        
        feature_importance = {}
        for i, emotion in enumerate(self.EMOTIONS):
            # Get top features for this emotion
            top_indices = np.argsort(coefficients[i])[-top_n:][::-1]
            top_features = [(feature_names[idx], coefficients[i][idx]) 
                          for idx in top_indices]
            feature_importance[emotion] = top_features
        
        return feature_importance


if __name__ == "__main__":
    # Example usage
    detector = EmotionDetector()
    
    # Example predictions
    test_texts = [
        "I am so happy and excited about this!",
        "This makes me really sad and depressed.",
        "I'm so angry about what happened!",
        "That movie was absolutely terrifying!",
        "Wow, I didn't expect that at all!",
        "This food tastes absolutely disgusting.",
        "The weather is okay today."
    ]
    
    print("Emotion Detection Examples:")
    print("-" * 40)
    
    for text in test_texts:
        # Note: This will only work after training the model
        print(f"Text: {text}")
        print(f"Preprocessing would give: {detector.preprocess_text(text)}")
        print()
