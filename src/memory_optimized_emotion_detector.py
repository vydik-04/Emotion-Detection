"""
Memory-Optimized Emotion Detection Model

This implementation focuses on three key memory optimization strategies:
1. Reducing the number of TF-IDF features
2. Using a smaller portion of the dataset for training
3. Implementing in-place computations to manage memory usage

Author: Memory-Optimized Version for Low-Resource Environments
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
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

class MemoryOptimizedEmotionModel:
    """
    Memory-optimized emotion detection model designed for environments 
    with limited computational resources.
    """
    
    def __init__(self, memory_mode='balanced'):
        """
        Initialize the memory-optimized emotion model.
        
        Args:
            memory_mode (str): 'aggressive', 'balanced', or 'conservative'
                - aggressive: Minimal memory usage, lower accuracy
                - balanced: Good balance between memory and accuracy  
                - conservative: Moderate memory savings, better accuracy
        """
        self.memory_mode = memory_mode
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Memory optimization settings based on mode
        self.config = self._get_memory_config()
        
        print(f"🔧 Memory optimization mode: {memory_mode}")
        print(f"📊 Configuration: {self.config}")
    
    def _get_memory_config(self):
        """Get memory optimization configuration based on mode."""
        configs = {
            'aggressive': {
                'max_features': 1000,          # Very low feature count
                'dataset_fraction': 0.05,      # Use only 5% of dataset
                'ngram_range': (1, 1),         # Only unigrams
                'batch_size': 1000,            # Small batch processing
                'cv_folds': 3,                 # Reduced CV folds
                'enable_smote': False,         # Disable SMOTE to save memory
                'n_estimators': 50             # Fewer trees in ensemble
            },
            'balanced': {
                'max_features': 2500,          # Moderate feature count
                'dataset_fraction': 0.1,       # Use 10% of dataset
                'ngram_range': (1, 2),         # Unigrams and bigrams
                'batch_size': 2000,            # Medium batch processing
                'cv_folds': 3,                 # Standard CV folds
                'enable_smote': True,          # Enable SMOTE with limits
                'n_estimators': 100            # Moderate ensemble size
            },
            'conservative': {
                'max_features': 5000,          # Higher feature count
                'dataset_fraction': 0.2,       # Use 20% of dataset
                'ngram_range': (1, 2),         # Unigrams and bigrams
                'batch_size': 5000,            # Larger batch processing
                'cv_folds': 5,                 # More CV folds
                'enable_smote': True,          # Enable full SMOTE
                'n_estimators': 150            # Larger ensemble
            }
        }
        return configs.get(self.memory_mode, configs['balanced'])
    
    def memory_efficient_text_preprocessing(self, texts, batch_size=None):
        """
        Memory-efficient text preprocessing using batch processing.
        
        Args:
            texts: List of texts to preprocess
            batch_size: Size of batches for processing (optional)
        
        Returns:
            List of preprocessed texts
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        print(f"🔄 Processing {len(texts)} texts in batches of {batch_size}...")
        
        processed_texts = []
        
        # Process in batches to save memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_processed = []
            
            for text in batch:
                processed_text = self._preprocess_single_text(text)
                batch_processed.append(processed_text)
            
            processed_texts.extend(batch_processed)
            
            # Save batch size before deletion
            current_batch_size = len(batch)
            
            # Force garbage collection after each batch
            del batch, batch_processed
            gc.collect()
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {i + current_batch_size} texts...")
        
        return processed_texts
    
    def _preprocess_single_text(self, text):
        """Preprocess a single text efficiently."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase (in-place where possible)
        text = str(text).lower()
        
        # Remove URLs and special characters (optimized regex)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Handle contractions efficiently
        contractions = {
            "n't": " not", "won't": " will not", "can't": " cannot"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Tokenize and filter (memory-efficient approach)
        tokens = word_tokenize(text)
        
        # Keep emotional words even if they're stopwords
        emotional_stopwords = {'not', 'no', 'never', 'very', 'really', 'so', 'too'}
        
        # Process tokens in-place
        filtered_tokens = []
        for token in tokens:
            if (token.isalpha() and len(token) > 2 and 
                (token not in self.stop_words or token in emotional_stopwords)):
                lemmatized = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(lemmatized)
        
        return ' '.join(filtered_tokens)
    
    def create_memory_efficient_vectorizer(self):
        """Create a memory-efficient TF-IDF vectorizer."""
        print(f"📊 Creating TF-IDF vectorizer with {self.config['max_features']} features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],  # Reduced feature count
            ngram_range=self.config['ngram_range'],    # Configurable n-grams
            analyzer='word',
            sublinear_tf=True,                         # Memory-efficient TF scaling
            min_df=3,                                  # Higher min_df to reduce features
            max_df=0.9,                                # Lower max_df to reduce features
            use_idf=True,
            smooth_idf=True,
            norm='l2',
            dtype=np.float32                           # Use float32 instead of float64
        )
        
        return self.vectorizer
    
    def memory_efficient_feature_extraction(self, texts):
        """Extract features with memory optimization."""
        print("🔧 Extracting features with memory optimization...")
        
        # Process texts in batches
        processed_texts = self.memory_efficient_text_preprocessing(texts)
        
        # Create vectorizer
        vectorizer = self.create_memory_efficient_vectorizer()
        
        # Transform texts to features
        print("🔄 Transforming texts to TF-IDF features...")
        tfidf_features = vectorizer.fit_transform(processed_texts)
        
        # Convert to dense array with memory consideration
        if tfidf_features.nnz / (tfidf_features.shape[0] * tfidf_features.shape[1]) < 0.1:
            # Keep sparse if less than 10% dense
            print("📊 Keeping sparse representation for memory efficiency")
            return tfidf_features
        else:
            # Convert to dense float32
            print("📊 Converting to dense float32 representation")
            return tfidf_features.toarray().astype(np.float32)
    
    def memory_efficient_smote(self, X, y):
        """Apply SMOTE with memory considerations."""
        if not self.config['enable_smote']:
            print("⚠️  SMOTE disabled in aggressive memory mode")
            return X, y
        
        print("🔄 Applying memory-efficient SMOTE...")
        
        # Convert sparse to dense if needed for SMOTE
        if hasattr(X, 'toarray'):
            X_dense = X.toarray().astype(np.float32)
        else:
            X_dense = X.astype(np.float32)
        
        # Apply SMOTE with memory limits
        smote = SMOTE(
            sampling_strategy='auto',
            random_state=42,
            k_neighbors=min(3, len(np.unique(y)) - 1)  # Adjust k_neighbors based on classes
        )
        
        try:
            X_balanced, y_balanced = smote.fit_resample(X_dense, y)
            
            # Clean up memory
            del X_dense
            gc.collect()
            
            print(f"📊 Balanced dataset: {len(X_balanced)} samples")
            return X_balanced.astype(np.float32), y_balanced
            
        except MemoryError:
            print("⚠️  Memory limit reached, skipping SMOTE")
            return X_dense, y
    
    def create_memory_efficient_model(self):
        """Create a memory-efficient ensemble model."""
        print("🤖 Creating memory-efficient ensemble model...")
        
        models = [
            ('logistic', LogisticRegression(
                C=0.5,  # Reduced regularization parameter
                max_iter=1000,  # Reduced iterations
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                n_jobs=1  # Single thread
            )),
            ('svm_linear', SVC(
                kernel='linear',
                C=0.5,  # Reduced C for faster training
                class_weight='balanced',
                probability=True,
                random_state=42
            )),
            ('random_forest', RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=10,  # Reduced depth
                min_samples_split=10,  # Increased to reduce overfitting
                min_samples_leaf=4,    # Increased to reduce memory
                class_weight='balanced',
                random_state=42,
                n_jobs=1  # Single thread to control memory
            ))
        ]
        
        # Create ensemble with soft voting
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=1  # Single thread
        )
        
        return ensemble
    
    def train_memory_efficient_model(self, dataset_path):
        """Train the model with memory optimizations."""
        print("🚀 Starting Memory-Optimized Model Training...")
        print("=" * 70)
        print(f"💾 Memory Mode: {self.memory_mode}")
        print(f"📊 Dataset Fraction: {self.config['dataset_fraction']}")
        print(f"🔧 Max Features: {self.config['max_features']}")
        print("=" * 70)
        
        # Load dataset with sampling
        print("📂 Loading and sampling dataset...")
        df_full = pd.read_csv(dataset_path)
        
        # Sample dataset based on memory mode
        sample_size = int(len(df_full) * self.config['dataset_fraction'])
        df = df_full.sample(n=sample_size, random_state=42)
        
        print(f"📊 Original dataset: {len(df_full)} samples")
        print(f"📊 Sampled dataset: {len(df)} samples ({self.config['dataset_fraction']*100:.1f}%)")
        
        # Clean up full dataset from memory
        del df_full
        gc.collect()
        
        # Display emotion distribution
        emotion_counts = df['emotion'].value_counts()
        print(f"📈 Emotion distribution (top 10):")
        for emotion, count in emotion_counts.head(10).items():
            print(f"   {emotion:20} | {count:5d} samples")
        
        # Extract features efficiently
        X = self.memory_efficient_feature_extraction(df['text'])
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['emotion'])
        
        # Clean up dataframe
        del df
        gc.collect()
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Memory usage: ~{X.nbytes / (1024**2):.1f} MB" if hasattr(X, 'nbytes') else "📊 Sparse matrix")
        
        # Apply SMOTE if enabled
        if self.config['enable_smote']:
            X, y = self.memory_efficient_smote(X, y)
        
        # Split data
        print("🔄 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Create and train model
        self.model = self.create_memory_efficient_model()
        
        print("🎯 Training model...")
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
        
        # Cross-validation with memory considerations
        print(f"\n🔄 Performing {self.config['cv_folds']}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=self.config['cv_folds'], 
            scoring='accuracy'
        )
        print(f"📊 CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Memory usage summary
        self._print_memory_summary()
        
        # Save model
        return self._save_memory_efficient_model(test_accuracy, cv_scores.mean())
    
    def _print_memory_summary(self):
        """Print memory usage summary."""
        print("\n💾 Memory Usage Summary:")
        print("-" * 50)
        print(f"🔧 TF-IDF Features: {self.config['max_features']}")
        print(f"📊 Dataset Fraction: {self.config['dataset_fraction']*100:.1f}%")
        print(f"🤖 Ensemble Size: {self.config['n_estimators']} trees")
        print(f"⚡ SMOTE Enabled: {'Yes' if self.config['enable_smote'] else 'No'}")
        print(f"🔄 CV Folds: {self.config['cv_folds']}")
    
    def _save_memory_efficient_model(self, test_accuracy, cv_accuracy):
        """Save the memory-efficient model and components."""
        model_dir = '../models'
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model filename based on memory mode
        model_name = f'memory_optimized_emotion_model_{self.memory_mode}'
        
        model_path = os.path.join(model_dir, f'{model_name}.joblib')
        vectorizer_path = os.path.join(model_dir, f'{model_name}_vectorizer.joblib')
        encoder_path = os.path.join(model_dir, f'{model_name}_encoder.joblib')
        
        # Save components
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"\n💾 Model saved to: {model_path}")
        print(f"💾 Vectorizer saved to: {vectorizer_path}")
        print(f"💾 Label encoder saved to: {encoder_path}")
        
        # Save configuration and results
        summary = {
            'memory_mode': self.memory_mode,
            'configuration': self.config,
            'test_accuracy': float(test_accuracy),
            'cv_accuracy': float(cv_accuracy),
            'feature_count': self.config['max_features'],
            'emotions': self.label_encoder.classes_.tolist()
        }
        
        summary_path = os.path.join(model_dir, f'{model_name}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved to: {summary_path}")
        
        return {
            'test_accuracy': test_accuracy,
            'cv_accuracy': cv_accuracy,
            'model_path': model_path,
            'summary': summary
        }
    
    def predict_memory_efficient(self, text):
        """Make prediction with memory efficiency."""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess single text
        processed_text = self._preprocess_single_text(text)
        
        # Transform to features
        features = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get emotion name
        emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get top probabilities
        top_emotions = {}
        for i, prob in enumerate(probabilities):
            emotion_name = self.label_encoder.inverse_transform([i])[0]
            top_emotions[emotion_name] = float(prob)
        
        return {
            'emotion': emotion,
            'confidence': float(max(probabilities)),
            'probabilities': top_emotions
        }

def compare_memory_modes():
    """Compare different memory optimization modes."""
    print("🔍 MEMORY OPTIMIZATION MODE COMPARISON")
    print("=" * 80)
    
    modes = ['aggressive', 'balanced', 'conservative']
    
    for mode in modes:
        print(f"\n📋 {mode.upper()} MODE:")
        print("-" * 40)
        
        model = MemoryOptimizedEmotionModel(memory_mode=mode)
        config = model.config
        
        # Estimate memory usage
        feature_memory = config['max_features'] * config['dataset_fraction'] * 4  # bytes per float32
        estimated_mb = feature_memory / (1024 * 1024)
        
        print(f"🔧 TF-IDF Features: {config['max_features']:,}")
        print(f"📊 Dataset Usage: {config['dataset_fraction']*100:.1f}%")
        print(f"💾 Est. Memory: ~{estimated_mb:.1f} MB")
        print(f"⚡ SMOTE: {'Enabled' if config['enable_smote'] else 'Disabled'}")
        print(f"🤖 Ensemble Size: {config['n_estimators']} estimators")

def main():
    """Main function to run memory-optimized training."""
    print("🚀 MEMORY-OPTIMIZED EMOTION DETECTION TRAINING")
    print("🎯 Optimized for Low-Resource Environments")
    print("=" * 80)
    
    # Show memory mode comparison
    compare_memory_modes()
    
    # Choose memory mode
    print("\n🔧 Select memory optimization mode:")
    print("1. Aggressive (Minimal memory, faster training)")
    print("2. Balanced (Good balance)")
    print("3. Conservative (Better accuracy, more memory)")
    
    try:
        choice = input("\nEnter choice (1-3) or press Enter for balanced: ").strip()
        mode_map = {'1': 'aggressive', '2': 'balanced', '3': 'conservative', '': 'balanced'}
        memory_mode = mode_map.get(choice, 'balanced')
    except:
        memory_mode = 'balanced'
    
    print(f"\n✅ Selected mode: {memory_mode}")
    
    # Initialize model
    model = MemoryOptimizedEmotionModel(memory_mode=memory_mode)
    
    # Train model
    dataset_path = '../data/emotion_dataset_1350k.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    try:
        results = model.train_memory_efficient_model(dataset_path)
        
        print("\n" + "=" * 80)
        print("🎉 MEMORY-OPTIMIZED TRAINING COMPLETED!")
        print("=" * 80)
        print(f"🎯 Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"🎯 CV Accuracy: {results['cv_accuracy']:.4f} ({results['cv_accuracy']*100:.2f}%)")
        print(f"💾 Memory Mode: {memory_mode}")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        test_text = "I am feeling very happy and excited about this!"
        prediction = model.predict_memory_efficient(test_text)
        print(f"Text: {test_text}")
        print(f"Predicted emotion: {prediction['emotion']} (confidence: {prediction['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
