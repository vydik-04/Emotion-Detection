# 🧠 Emotion Detection from Text Using Traditional ML (27 Emotions)

A powerful **Emotion Detection from Text** system that combines **Machine Learning + Rule-Based NLP** to accurately classify text into **27 different emotions**.

This project goes beyond traditional models by implementing a **hybrid architecture**, improving accuracy, handling negations, and optimizing performance for real-world usage.

---

## 🚀 Features

  - 🔥 Detects **27 different emotions** (GoEmotions-based)
  - ⚡ Hybrid system (ML + Rule-based)
  - 🧠 Advanced NLP preprocessing (negation handling, lemmatization)
  - 🤖 Ensemble Machine Learning models
  - 📊 Confidence scores & top emotion predictions
  - 💾 Memory-optimized model for low-resource systems
  - 🧪 Extensive testing & evaluation support

---

## 🏗️ Project Architecture


User Input Text
↓
Text Preprocessing
↓
Feature Extraction (TF-IDF)
↓
├── ML Model (Ensemble)
└── Rule-Based Engine
↓
Hybrid Decision Engine
↓
Final Emotion Prediction


---

## 📂 Project Structure


├── data/
│ ├── emotion_dataset_1350k.csv
│ ├── massive_emotion_dataset.csv
│ └── emotion_dataset_augmented.csv

├── models/
│ ├── ultra_high_accuracy_emotion_model.joblib
│ ├── memory_optimized_emotion_model_*.joblib
│ ├── vectorizer.joblib
│ └── label_encoder.joblib

├── src/
│ ├── emotion_detector.py
│ ├── ultimate_emotion_detector.py
│ ├── memory_optimized_emotion_detector.py

├── training/
│ ├── train_ultra_high_accuracy.py
│ ├── improve_model_accuracy.py
│ └── evaluate_27_emotions.py

├── rule_based/
│ ├── emotion_keywords.json
│ └── emotion_keywords_complete.json

├── debugging/
│ ├── debug_patterns.py
│ ├── debug_strength.py
│ └── debug_classes.py

├── tests/
│ ├── test_negation.py
│ ├── test_sadness.py
│ └── quick_test.py

├── app.py
├── requirements.txt
└── README.md


---

## 🧠 How It Works

### 1. Data Collection
  - Uses large-scale datasets like:
    - GoEmotions (27 emotions)
    - Custom augmented datasets
  - ~1.35 million samples (balanced across emotions)

---

### 2. Text Preprocessing
  - Lowercasing
  - Removing URLs & noise
  - Tokenization
  - Lemmatization
  - **Negation handling (important)**  

Example:
  "not happy" → sadness (not joy)
---

### 3. Feature Engineering
  - TF-IDF Vectorization
  - N-grams (1–3)
  - Converts text into numerical vectors

---

### 4. Machine Learning Models
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - Naive Bayes

👉 Combined using:
  **Voting Classifier (Soft Voting Ensemble)**
---

### 5. Rule-Based System
  - Keyword matching
  - Regex patterns
  - Emotion-specific rules

Example:
  "I hate this" → anger
  "I am not happy" → sadness
---

### 6. Hybrid Decision Engine
Combines:
  - ML prediction (probabilities)
  - Rule-based scores

Logic:
  - Strong rule → override ML
  - Else → use ML prediction
---

### 7. Output:
  - Final predicted emotion
  - Confidence score
  - Top-N emotions (optional)

---

## 📊 Model Performance:

  | Model Version              | Accuracy |
  |--------------------------|---------|
  | Initial Model            | ~50%    |
  | Improved (100K data)     | ~74%    |
  | Optimized Models         | ~98–99% (small dataset) |
  | Memory Optimized Model   | ~97–98% |

> Note: Accuracy varies based on dataset size and class complexity (27 classes is challenging).
---

## ⚡ Memory Optimization:
The project includes a **memory-optimized model** with 3 modes:
  - `aggressive` → Low memory, faster
  - `balanced` → Best trade-off
  - `conservative` → Higher accuracy

Techniques used:
  - Reduced TF-IDF features
  - Batch processing
  - Sparse matrices
  - Limited dataset sampling
---

## 🧪 Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
---

## ▶️ How to Run:
### 1. Install dependencies
  ```bash
  pip install -r requirements.txt
  2. Run the app
  python app.py
  3. Test prediction
  python quick_test.py

📌 Example:

  from emotion_detector import EmotionDetector
  detector = EmotionDetector()
  text = "I am feeling very happy today!"
  result = detector.predict(text)
  print(result)

Output:
  {
    "emotion": "joy",
    "confidence": 0.92
  }

🔍 Key Highlights:
✔  Hybrid ML + Rule-Based system
✔  Handles negation (major NLP challenge)
✔  Works on 27 emotion classes
✔  Scalable & optimized
✔  Production-ready design

⚠️ Known Issues:
  Dataset imbalance affects rare emotions
  Some classes (e.g., awe, relief) have low performance
  Augmentation pipeline needs refinement
  🚀 Future Improvements
  Use BERT / Transformers for better accuracy
  Improve rare class prediction
  Deploy as API (Flask/FastAPI)
  Add real-time UI dashboard

⭐ If you like this project:
  Give it a ⭐ on GitHub and share it!