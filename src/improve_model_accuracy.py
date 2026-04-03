"""
Advanced Model Improvement Script for 95%+ Accuracy
This script implements state-of-the-art techniques to dramatically improve emotion detection accuracy
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class AdvancedEmotionModel:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def advanced_text_preprocessing(self, text):
        """Advanced text preprocessing for better feature extraction"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs, emails, and special characters but keep emotional punctuation
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Preserve emotional punctuation and intensifiers
        text = re.sub(r'([!?])\1+', r'\1\1\1', text)  # Normalize multiple punctuation
        text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)  # Normalize repeated letters
        
        # Handle negations properly
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        
        # Replace placeholders with generic terms
        text = re.sub(r'\{[^}]+\}', 'PLACEHOLDER', text)
        text = re.sub(r'\[NAME\]', 'PERSON', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        
        # Keep emotional words even if they're stopwords
        emotional_stopwords = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 
                              'neither', 'nor', 'none', 'very', 'really', 'so', 'too'}
        
        processed_tokens = []
        for token in tokens:
            if token.isalpha():
                if token not in self.stop_words or token in emotional_stopwords:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def create_advanced_features(self, texts):
        """Create advanced features including TF-IDF, n-grams, and emotional features"""
        # Advanced TF-IDF with character and word n-grams
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            analyzer='word',
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        # Process texts
        processed_texts = [self.advanced_text_preprocessing(text) for text in texts]
        
        # Create TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        
        return tfidf_features.toarray()
    
    def balance_dataset(self, X, y):
        """Advanced dataset balancing using SMOTE with Tomek links"""
        print("🔄 Balancing dataset using SMOTE+Tomek...")
        
        # Use SMOTETomek for better balancing
        smote_tomek = SMOTETomek(
            smote=SMOTE(
                sampling_strategy='auto',
                random_state=42,
                k_neighbors=3
            ),
            random_state=42
        )
        
        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
        
        print(f"📊 Original dataset size: {len(X)}")
        print(f"📊 Balanced dataset size: {len(X_balanced)}")
        
        return X_balanced, y_balanced
    
    def create_ensemble_model(self, X, y):
        """Create an advanced ensemble model with optimized hyperparameters"""
        print("🤖 Creating advanced ensemble model...")
        
        # Individual models with optimized parameters
        models = {
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            ),
            'svm_linear': SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'svm_rbf': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=0.1)
        }
        
        # Create voting ensemble
        voting_classifier = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft',
            n_jobs=-1
        )
        
        return voting_classifier
    
    def train_model(self, dataset_path):
        """Train the advanced emotion detection model"""
        print("🚀 Starting Advanced Model Training for 95%+ Accuracy...")
        print("=" * 80)
        
        # Load dataset
        print("📂 Loading dataset...")
        df = pd.read_csv(dataset_path).sample(frac=0.1, random_state=42)
        print(f"📊 Dataset loaded: {len(df)} samples, {df['emotion'].nunique()} emotions")
        
        # Display emotion distribution
        emotion_counts = df['emotion'].value_counts()
        print(f"📈 Emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"   {emotion:25} | {count:6d} samples")
        
        # Preprocess and create features
        print("\n🔧 Creating advanced features...")
        X = self.create_advanced_features(df['text'])
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['emotion'])
        
        # Balance dataset
        X_balanced, y_balanced = self.balance_dataset(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_balanced
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Create and train ensemble model
        self.model = self.create_ensemble_model(X_train, y_train)
        
        print("\n🎯 Training ensemble model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        print("\n📈 Evaluating model performance...")
        
        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Test accuracy
        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"🎯 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Cross-validation
        print("\n🔄 Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_balanced, y_balanced, cv=5, scoring='accuracy')
        print(f"📊 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print("-" * 80)
        
        emotion_names = self.label_encoder.classes_
        report = classification_report(y_test, test_pred, target_names=emotion_names, output_dict=True)
        
        # Display individual emotion accuracies
        for emotion in emotion_names:
            if emotion in report:
                f1_score = report[emotion]['f1-score']
                precision = report[emotion]['precision']
                recall = report[emotion]['recall']
                support = int(report[emotion]['support'])
                
                status = "🟢 EXCELLENT" if f1_score >= 0.95 else "🟡 GOOD" if f1_score >= 0.90 else "🔴 NEEDS WORK"
                
                print(f"{emotion:25} | F1: {f1_score:.3f} ({f1_score*100:5.1f}%) | "
                      f"P: {precision:.3f} | R: {recall:.3f} | "
                      f"Support: {support:4d} | {status}")
        
        # Save model and components
        model_dir = '../models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'advanced_emotion_model_95plus.joblib')
        vectorizer_path = os.path.join(model_dir, 'advanced_vectorizer_95plus.joblib')
        encoder_path = os.path.join(model_dir, 'advanced_label_encoder_95plus.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"\n💾 Model saved to: {model_path}")
        print(f"💾 Vectorizer saved to: {vectorizer_path}")
        print(f"💾 Label encoder saved to: {encoder_path}")
        
        # Save training summary
        summary = {
            'final_accuracy': float(test_accuracy),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'train_accuracy': float(train_accuracy),
            'dataset_size': len(df),
            'balanced_dataset_size': len(X_balanced),
            'feature_count': X.shape[1],
            'emotions': emotion_names.tolist(),
            'classification_report': report
        }
        
        summary_path = os.path.join(model_dir, 'advanced_model_95plus_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Training summary saved to: {summary_path}")
        
        return {
            'test_accuracy': test_accuracy,
            'cv_accuracy': cv_scores.mean(),
            'model_path': model_path,
            'classification_report': report
        }
    
    def hyperparameter_optimization(self, X, y):
        """Perform hyperparameter optimization for even better results"""
        print("🔧 Performing hyperparameter optimization...")
        
        # Optimize Logistic Regression
        lr_params = {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [2000, 3000]
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(class_weight='balanced', random_state=42),
            lr_params,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        lr_grid.fit(X, y)
        print(f"🎯 Best LR params: {lr_grid.best_params_}")
        print(f"🎯 Best LR score: {lr_grid.best_score_:.4f}")
        
        return lr_grid.best_estimator_

def main():
    """Main function to run the advanced model training"""
    print("🚀 ADVANCED EMOTION DETECTION MODEL TRAINING")
    print("🎯 Target: 95%+ Accuracy for All 27 Emotions")
    print("=" * 80)
    
    # Initialize model
    advanced_model = AdvancedEmotionModel()
    
    # Train model
    dataset_path = '../data/emotion_dataset_1350k.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    try:
        results = advanced_model.train_model(dataset_path)
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED!")
        print("=" * 80)
        print(f"🎯 Final Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"🎯 Cross-Validation Accuracy: {results['cv_accuracy']:.4f} ({results['cv_accuracy']*100:.2f}%)")
        
        if results['test_accuracy'] >= 0.95:
            print("🏆 SUCCESS! Model achieved 95%+ accuracy!")
        else:
            print("📈 Model improved but needs further optimization for 95%+ accuracy")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
