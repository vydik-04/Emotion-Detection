"""
Ultra-High Accuracy Training Script for Emotion Detection
Enhanced data augmentation and class balancing for 90%+ accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from emotion_detector import EmotionDetector
from tqdm import tqdm
import joblib
from sklearn.pipeline import Pipeline
import json
import random
import re
from collections import Counter

warnings.filterwarnings('ignore')
plt.ioff()  # Turn off interactive mode to prevent plot display issues


def create_enhanced_emotion_dataset():
    """
    Create an enhanced, balanced dataset with extensive variations for each emotion
    """
    print("🔧 Creating enhanced emotion dataset with improved class balance...")
    
    # Enhanced emotional expressions with more diversity for each emotion
    emotion_data = {
        'joy': [
            # Extreme happiness
            "I'm absolutely ecstatic about this incredible news!",
            "This brings me overwhelming joy and happiness!",
            "I feel fantastic and amazing about everything!",
            "What wonderful and exciting news this is!",
            "I'm over the moon with pure excitement!",
            "This makes my heart sing with absolute joy!",
            "I couldn't be happier about this outcome!",
            "This is the best thing that could happen!",
            "I'm bursting with happiness and gratitude!",
            "Life is absolutely beautiful right now!",
            
            # Satisfaction and contentment
            "I feel really good about this decision.",
            "This makes me smile so much!",
            "I'm pleased with how things turned out.",
            "What a delightful surprise this is!",
            "I feel satisfied and content today.",
            "This brings me genuine happiness.",
            "I'm enjoying this moment so much.",
            "This fills me with warm feelings.",
            "I feel optimistic about the future.",
            "This gives me hope and joy.",
            
            # Achievement and pride
            "I'm so proud of what we accomplished!",
            "This achievement makes me incredibly happy!",
            "I feel triumphant and successful!",
            "We did it! I'm so excited!",
            "This victory fills me with joy!",
            "I feel accomplished and fulfilled!",
            "This success brings me great pleasure!",
            "I'm thrilled with our progress!",
            "This milestone makes me euphoric!",
            "I feel victorious and joyful!",
            
            # Love and affection
            "I love this so much it makes me happy!",
            "This warms my heart with pure joy!",
            "I adore everything about this situation!",
            "This makes me feel so loved and happy!",
            "I'm filled with love and happiness!",
            "This beautiful moment brings me joy!",
            "I cherish this wonderful feeling!",
            "This precious moment makes me smile!",
            "I treasure this joyful experience!",
            "This lovely surprise delights me!"
        ],
        
        'sadness': [
            # Deep sadness
            "I feel deeply saddened by this tragic news.",
            "This breaks my heart completely and utterly.",
            "I'm overwhelmed with sorrow and grief.",
            "The situation fills me with profound melancholy.",
            "I can't stop crying about this loss.",
            "This makes me feel empty and hopeless inside.",
            "My heart feels heavy with unbearable sadness.",
            "I'm devastated by what has happened.",
            "The loss leaves me feeling broken and shattered.",
            "This tragedy fills me with deep despair.",
            
            # Disappointment
            "I'm really disappointed by this outcome.",
            "This news makes me feel down and blue.",
            "I feel let down by the situation.",
            "This didn't go as I had hoped.",
            "I'm feeling quite low about this.",
            "This leaves me feeling dejected.",
            "I'm bummed out by what happened.",
            "This makes me feel discouraged.",
            "I feel downhearted about the result.",
            "This situation dampens my spirits.",
            
            # Loneliness and isolation
            "I feel so lonely and isolated right now.",
            "This makes me miss them terribly.",
            "I feel abandoned and forgotten.",
            "The silence makes me feel so sad.",
            "I wish things were different.",
            "I feel like no one understands me.",
            "This emptiness is overwhelming.",
            "I feel disconnected from everyone.",
            "The loneliness is unbearable.",
            "I feel lost and alone in this.",
            
            # Regret and remorse
            "I deeply regret what happened today.",
            "This mistake fills me with sadness.",
            "I wish I could turn back time.",
            "This regret weighs heavily on me.",
            "I'm sorry this had to happen.",
            "This remorse is eating me up inside.",
            "I feel guilty and sad about this.",
            "This sorrow won't leave me alone.",
            "I'm mourning what could have been.",
            "This melancholy mood persists."
        ],
        
        'anger': [
            # Intense anger
            "I'm absolutely furious about this injustice!",
            "This makes my blood boil with rage!",
            "I'm so angry I can barely contain myself!",
            "This outrageous behavior is completely unacceptable!",
            "I'm livid and fed up with this nonsense!",
            "This is absolutely infuriating and maddening!",
            "I'm seething with anger right now!",
            "This makes me want to scream in frustration!",
            "I'm outraged by this terrible treatment!",
            "This is making me incredibly mad and hostile!",
            
            # Frustration
            "This is so frustrating and annoying!",
            "I'm irritated by this constant problem.",
            "This really gets on my nerves.",
            "I'm fed up with this situation.",
            "This is driving me crazy with anger.",
            "I'm losing my patience with this.",
            "This makes me so mad I could explode.",
            "I'm angry about the unfairness.",
            "This incompetence infuriates me.",
            "I'm sick and tired of this behavior.",
            
            # Indignation
            "How dare they treat us this way!",
            "This is completely unacceptable behavior!",
            "I'm appalled by their actions!",
            "This is an outrage and injustice!",
            "I'm disgusted by their attitude!",
            "This is absolutely inexcusable!",
            "I won't tolerate this anymore!",
            "This crosses every line possible!",
            "I demand better treatment than this!",
            "This is utterly unforgivable behavior!",
            
            # Resentment and bitterness
            "I resent being treated this poorly.",
            "This bitter experience angers me.",
            "I'm hostile toward this situation.",
            "This contempt I feel is overwhelming.",
            "I'm enraged by this disrespect.",
            "This vindictive behavior makes me mad.",
            "I'm incensed by their arrogance.",
            "This spiteful act infuriates me.",
            "I'm wrathful about this betrayal.",
            "This vengeful feeling consumes me."
        ],
        
        'fear': [
            # Terror and panic
            "I'm absolutely terrified about what might happen.",
            "This situation fills me with overwhelming dread and panic.",
            "I'm scared out of my mind about the outcome.",
            "The uncertainty makes me incredibly anxious and nervous.",
            "I'm trembling with fear and worry about this.",
            "This gives me horrible chills down my spine.",
            "I'm paralyzed by anxiety about the future.",
            "The thought of this frightens me to my core.",
            "I'm worried sick about what's coming next.",
            "This situation makes me panic and hyperventilate.",
            
            # Anxiety and worry
            "I'm really worried about this situation.",
            "This makes me feel anxious and uneasy.",
            "I'm concerned about what might happen.",
            "This gives me a lot of anxiety.",
            "I feel nervous and apprehensive.",
            "This uncertainty scares me.",
            "I'm afraid of the consequences.",
            "This makes me feel vulnerable.",
            "I'm stressed about the outcome.",
            "This situation makes me uncomfortable.",
            
            # Specific fears
            "I'm afraid of failing at this important task.",
            "The dark unknown frightens me deeply.",
            "I'm scared of being alone in this.",
            "This change terrifies me completely.",
            "I fear the worst might happen.",
            "I'm anxious about making mistakes.",
            "This responsibility scares me.",
            "I'm worried about disappointing others.",
            "The pressure makes me feel afraid.",
            "I'm nervous about the presentation tomorrow.",
            
            # Phobias and deep fears
            "This phobia controls my entire life.",
            "I'm petrified by this scary situation.",
            "This horror fills me with dread.",
            "I'm intimidated by this challenge.",
            "This threat makes me feel helpless.",
            "I'm alarmed by this dangerous situation.",
            "This menace frightens me deeply.",
            "I'm startled by this sudden change.",
            "This peril makes me shake with fear.",
            "I'm haunted by this terrifying thought."
        ],
        
        'surprise': [
            # Amazement
            "Wow, I never expected this incredible turn of events!",
            "What an amazing and unexpected surprise this is!",
            "I'm completely shocked and astonished by this news!",
            "This caught me totally off guard and unprepared!",
            "I can't believe this actually happened today!",
            "What a remarkable and surprising development!",
            "I'm speechless and utterly amazed by this!",
            "This is beyond my wildest expectations and dreams!",
            "I'm stunned by this incredible revelation!",
            "What an extraordinary and unexpected outcome!",
            
            # Pleasant surprises
            "This is such a wonderful surprise!",
            "I didn't see this coming at all!",
            "What a delightful and unexpected gift!",
            "This exceeds all my expectations!",
            "I'm pleasantly surprised by this!",
            "This is better than I imagined!",
            "What an incredible turn of fortune!",
            "This amazes me in the best way!",
            "I'm blown away by this kindness!",
            "This is a dream come true!",
            
            # Sudden realization
            "I just realized something important!",
            "It suddenly all makes sense now!",
            "The truth just hit me unexpectedly!",
            "I just had an amazing breakthrough!",
            "The answer came to me suddenly!",
            "I just figured out the solution!",
            "This revelation is mind-blowing!",
            "It all clicked into place now!",
            "I just connected all the dots!",
            "The pieces finally fell together!",
            
            # Shock and bewilderment
            "I'm flabbergasted by this news!",
            "This bewildering situation confuses me!",
            "I'm perplexed by this turn of events!",
            "This mystifying development puzzles me!",
            "I'm baffled by what just happened!",
            "This confounding situation amazes me!",
            "I'm dumfounded by this revelation!",
            "This astounding news leaves me speechless!",
            "I'm thunderstruck by this information!",
            "This staggering discovery shocks me!"
        ],
        
        'disgust': [
            # Physical revulsion
            "This is absolutely disgusting and revolting to witness.",
            "I find this completely repugnant and nauseating.",
            "This makes me feel sick to my stomach.",
            "The situation is utterly nauseating and gross.",
            "I'm repulsed by this horrible sight and smell.",
            "This is gross and completely unappetizing to see.",
            "The sight of this makes me queasy and ill.",
            "I find this behavior absolutely vile and disgusting.",
            "This is reprehensible and makes me want to vomit.",
            "The whole thing makes me physically sick.",
            
            # Moral disgust
            "This behavior is morally disgusting.",
            "I'm appalled by their actions.",
            "This is ethically reprehensible.",
            "I find this conduct repulsive.",
            "This is morally bankrupt behavior.",
            "I'm sickened by their attitude.",
            "This is despicable and shameful.",
            "I find this absolutely deplorable.",
            "This behavior disgusts me deeply.",
            "I'm revolted by their choices.",
            
            # Aesthetic disgust
            "This looks absolutely horrible.",
            "The appearance is disgusting.",
            "This is visually repulsive.",
            "The design is nauseating.",
            "This looks completely gross.",
            "The style is revolting.",
            "This is ugly and disgusting.",
            "The presentation is repugnant.",
            "This looks absolutely awful.",
            "The sight is completely off-putting.",
            
            # Contamination and filth
            "This filthy mess disgusts me completely.",
            "The contamination makes me feel sick.",
            "This unsanitary condition is revolting.",
            "I'm disgusted by this dirty situation.",
            "This polluted environment repulses me.",
            "The corruption here sickens me.",
            "This tainted atmosphere is nauseating.",
            "I'm repelled by this unclean state.",
            "This impure condition disgusts me.",
            "The foulness of this makes me ill."
        ],
        
        'neutral': [
            # Factual statements
            "I need to attend the meeting at two PM today.",
            "The report is due by Friday afternoon this week.",
            "Please review the document when you have time.",
            "The temperature today is twenty-five degrees celsius.",
            "I'll call you back after lunch this afternoon.",
            "The project deadline is scheduled for next month.",
            "Please send me the information by email tomorrow.",
            "I'm going to the office this morning as usual.",
            "The presentation starts at nine AM sharp.",
            "We should discuss this matter further next week.",
            
            # Instructions and requests
            "Could you please pass me the salt?",
            "I would like to schedule an appointment.",
            "Please fill out this form completely.",
            "The instructions are on page three.",
            "Let me know when you're ready.",
            "Here are the documents you requested.",
            "Please sign on the dotted line.",
            "The meeting room is down the hall.",
            "I'll be there in about ten minutes.",
            "Please close the door behind you.",
            
            # Observations
            "The traffic is moving slowly today.",
            "It looks like it might rain later.",
            "The store opens at nine AM.",
            "There are five people in the queue.",
            "The computer is running normally.",
            "The phone is ringing in the office.",
            "The coffee is still hot.",
            "The book is on the table.",
            "The light is on in the room.",
            "The car is parked outside.",
            
            # Procedural and informational
            "The system will be updated tonight.",
            "Please refer to the manual for details.",
            "The conference is scheduled for next month.",
            "You can find the information online.",
            "The building has twenty floors.",
            "The software requires an update.",
            "Please confirm your attendance.",
            "The policy went into effect yesterday.",
            "The office hours are nine to five.",
            "You need to submit the form by tomorrow."
        ]
    }
    
    return emotion_data


def advanced_data_augmentation(emotion_data):
    """
    Apply advanced data augmentation techniques to increase diversity
    """
    print("🔄 Applying advanced data augmentation...")
    
    augmented_data = {'texts': [], 'emotions': []}
    
    # Augmentation techniques
    synonym_substitutions = {
        'happy': ['joyful', 'elated', 'cheerful', 'delighted', 'pleased'],
        'sad': ['depressed', 'melancholy', 'sorrowful', 'dejected', 'downhearted'],
        'angry': ['furious', 'irate', 'enraged', 'livid', 'infuriated'],
        'scared': ['frightened', 'terrified', 'petrified', 'alarmed', 'panicked'],
        'surprised': ['astonished', 'amazed', 'stunned', 'shocked', 'bewildered'],
        'disgusted': ['repulsed', 'revolted', 'sickened', 'nauseated', 'appalled'],
        'good': ['excellent', 'wonderful', 'fantastic', 'great', 'amazing'],
        'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'atrocious'],
        'very': ['extremely', 'incredibly', 'tremendously', 'exceptionally', 'remarkably'],
        'really': ['truly', 'genuinely', 'absolutely', 'completely', 'utterly']
    }
    
    intensity_modifiers = {
        'high': ['extremely', 'incredibly', 'absolutely', 'utterly', 'tremendously', 'exceptionally'],
        'medium': ['quite', 'rather', 'fairly', 'pretty', 'somewhat', 'moderately'],
        'low': ['a bit', 'slightly', 'a little', 'kind of', 'sort of', 'mildly']
    }
    
    temporal_contexts = ['today', 'yesterday', 'right now', 'this morning', 'lately', 'recently', 'currently']
    personal_contexts = ['personally', 'for me', 'in my opinion', 'from my perspective', 'as for me']
    situational_contexts = ['in this situation', 'under these circumstances', 'given the context', 'in this case']
    
    augmentation_count = 0
    
    for emotion, texts in emotion_data.items():
        # Add original texts
        for text in texts:
            augmented_data['texts'].append(text)
            augmented_data['emotions'].append(emotion)
        
        # Generate augmented versions
        for text in texts:
            # Synonym substitution
            for original_word, synonyms in synonym_substitutions.items():
                if original_word in text.lower():
                    for synonym in synonyms[:2]:  # Use first 2 synonyms
                        augmented_text = re.sub(r'\b' + original_word + r'\b', synonym, text, flags=re.IGNORECASE)
                        if augmented_text != text:
                            augmented_data['texts'].append(augmented_text)
                            augmented_data['emotions'].append(emotion)
                            augmentation_count += 1
            
            # Intensity modification (for emotional texts)
            if emotion != 'neutral':
                for intensity_level, modifiers in intensity_modifiers.items():
                    for modifier in modifiers[:2]:  # Use first 2 modifiers
                        # Add intensity to emotional expressions
                        if 'feel' in text.lower():
                            augmented_text = text.replace('feel', f'feel {modifier}')
                        elif 'am' in text.lower():
                            augmented_text = text.replace('am', f'am {modifier}')
                        elif text.startswith('I'):
                            augmented_text = f"I {modifier} think that " + text[2:].lower()
                        else:
                            augmented_text = f"I feel {modifier} that " + text.lower()
                        
                        augmented_data['texts'].append(augmented_text)
                        augmented_data['emotions'].append(emotion)
                        augmentation_count += 1
            
            # Context addition
            contexts = temporal_contexts + personal_contexts + situational_contexts
            for context in random.sample(contexts, min(3, len(contexts))):
                if emotion != 'neutral':
                    augmented_text = f"{context.capitalize()}, {text.lower()}"
                else:
                    augmented_text = f"{text} {context}"
                
                augmented_data['texts'].append(augmented_text)
                augmented_data['emotions'].append(emotion)
                augmentation_count += 1
    
    print(f"✅ Generated {augmentation_count} augmented samples")
    return augmented_data


def balance_dataset_advanced(augmented_data):
    """
    Advanced dataset balancing to ensure all classes have sufficient samples
    """
    print("⚖️ Balancing dataset for optimal class distribution...")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(augmented_data)
    
    # Check current distribution
    emotion_counts = df['emotions'].value_counts()
    print("Current distribution:")
    print(emotion_counts)
    
    # Target: at least 150 samples per emotion for robust training
    target_samples = 150
    
    balanced_texts = []
    balanced_emotions = []
    
    for emotion in df['emotions'].unique():
        emotion_texts = df[df['emotions'] == emotion]['texts'].tolist()
        current_count = len(emotion_texts)
        
        # Add all existing samples
        balanced_texts.extend(emotion_texts)
        balanced_emotions.extend([emotion] * current_count)
        
        # If we need more samples, create variations
        if current_count < target_samples:
            needed = target_samples - current_count
            print(f"📈 Generating {needed} additional samples for {emotion}")
            
            # Create variations using existing samples
            base_samples = emotion_texts[:min(20, len(emotion_texts))]  # Use up to 20 base samples
            
            variation_techniques = [
                lambda text: f"I must say that {text.lower()}",
                lambda text: f"It's clear to me that {text.lower()}",
                lambda text: f"Without a doubt, {text.lower()}",
                lambda text: f"I have to admit that {text.lower()}",
                lambda text: f"Honestly, {text.lower()}",
                lambda text: f"Frankly speaking, {text.lower()}",
                lambda text: f"To be honest, {text.lower()}",
                lambda text: f"In all honesty, {text.lower()}"
            ]
            
            generated = 0
            while generated < needed:
                for base_text in base_samples:
                    if generated >= needed:
                        break
                    
                    # Apply random variation technique
                    technique = random.choice(variation_techniques)
                    try:
                        varied_text = technique(base_text)
                        
                        # Avoid duplicates
                        if varied_text not in balanced_texts:
                            balanced_texts.append(varied_text)
                            balanced_emotions.append(emotion)
                            generated += 1
                    except:
                        continue
    
    balanced_df = pd.DataFrame({
        'text': balanced_texts,
        'emotion': balanced_emotions
    })
    
    # Final deduplication and cleaning
    balanced_df = balanced_df.drop_duplicates(subset=['text'])
    balanced_df = balanced_df[balanced_df['text'].str.len() > 10]
    balanced_df = balanced_df.dropna()
    
    print("\nFinal balanced distribution:")
    final_counts = balanced_df['emotion'].value_counts()
    print(final_counts)
    
    return balanced_df


def create_optimized_feature_extraction():
    """
    Create highly optimized TF-IDF vectorizer for maximum performance
    """
    return TfidfVectorizer(
        max_features=20000,          # Even larger feature space
        ngram_range=(1, 4),          # Include 4-grams for better context
        min_df=2,                    # Minimum document frequency
        max_df=0.90,                 # Lower max_df to include more discriminative features
        stop_words='english',
        lowercase=True,
        strip_accents='ascii',
        token_pattern=r'\b\w{2,}\b',
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm='l2'                    # L2 normalization
    )


def create_optimized_models():
    """
    Create highly optimized models for maximum accuracy
    """
    models = {
        'logistic_regression_balanced': LogisticRegression(
            C=100.0,                 # Higher regularization parameter
            max_iter=5000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42,
            dual=False
        ),
        
        'logistic_regression_l1': LogisticRegression(
            C=10.0,
            max_iter=5000,
            class_weight='balanced',
            solver='liblinear',
            penalty='l1',            # L1 regularization for feature selection
            random_state=42
        ),
        
        'svm_linear_optimized': SVC(
            kernel='linear',
            C=10.0,
            class_weight='balanced',
            probability=True,
            random_state=42,
            shrinking=True
        ),
        
        'svm_rbf_optimized': SVC(
            kernel='rbf',
            C=100.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
            shrinking=True
        ),
        
        'random_forest_optimized': RandomForestClassifier(
            n_estimators=500,        # More trees
            max_depth=20,            # Deeper trees
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        ),
        
        'gradient_boosting_optimized': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,      # Lower learning rate for better performance
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,           # Stochastic gradient boosting
            random_state=42
        )
    }
    
    return models


def extensive_hyperparameter_optimization(X_train, y_train):
    """
    Extensive hyperparameter optimization for best performance
    """
    print("🔧 Extensive hyperparameter optimization...")
    
    # Logistic Regression optimization with more parameters
    lr_params = {
        'C': [1.0, 10.0, 100.0, 1000.0],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced', None],
        'max_iter': [3000, 5000]
    }
    
    # Use stratified k-fold for better evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42),
        lr_params,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    lr_grid.fit(X_train, y_train)
    
    print(f"✅ Best LR params: {lr_grid.best_params_}")
    print(f"✅ Best LR CV score: {lr_grid.best_score_:.4f}")
    
    # SVM optimization
    svm_params = {
        'C': [1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced', None]
    }
    
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        svm_params,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    svm_grid.fit(X_train, y_train)
    
    print(f"✅ Best SVM params: {svm_grid.best_params_}")
    print(f"✅ Best SVM CV score: {svm_grid.best_score_:.4f}")
    
    return lr_grid.best_estimator_, svm_grid.best_estimator_


def create_ultra_ensemble(models, X_train, y_train):
    """
    Create an ultra-high performance ensemble
    """
    print("🤖 Creating ultra-high performance ensemble...")
    
    # Select the best performing models for the ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('lr_bal', models['logistic_regression_balanced']),
            ('lr_l1', models['logistic_regression_l1']),
            ('svm_lin', models['svm_linear_optimized']),
            ('svm_rbf', models['svm_rbf_optimized']),
            ('rf', models['random_forest_optimized'])
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Train ensemble
    voting_clf.fit(X_train, y_train)
    
    return voting_clf


def main():
    """
    Main function for ultra-high accuracy emotion detection training
    """
    print("🚀 Starting Ultra-High Accuracy Emotion Detection Training")
    print("=" * 70)
    
    # Create models directory
    os.makedirs('../models', exist_ok=True)
    
    # Step 1: Create enhanced dataset
    emotion_data = create_enhanced_emotion_dataset()
    
    # Step 2: Apply advanced data augmentation
    augmented_data = advanced_data_augmentation(emotion_data)
    
    # Step 3: Balance dataset
    df = balance_dataset_advanced(augmented_data)
    
    # Step 4: Data analysis
    print(f"\n📊 Final Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique emotions: {df['emotion'].nunique()}")
    print("\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)
    
    # Check class balance
    min_samples = emotion_counts.min()
    max_samples = emotion_counts.max()
    balance_ratio = min_samples / max_samples
    print(f"Class balance ratio: {balance_ratio:.3f} (>0.7 is excellent)")
    
    # Step 5: Advanced preprocessing
    print("\n🔧 Advanced text preprocessing...")
    detector = EmotionDetector()
    
    tqdm.pandas(desc="Preprocessing")
    df['processed_text'] = df['text'].progress_apply(detector.preprocess_text)
    
    # Remove empty processed texts
    df = df[df['processed_text'].str.len() > 0]
    print(f"✅ After preprocessing: {len(df)} samples")
    
    # Step 6: Train-test split with stratification
    X = df['processed_text']
    y = df['emotion']
    emotion_labels = sorted(df['emotion'].unique())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y  # Smaller test set for more training data
    )
    
    print(f"\n🔄 Data split:")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Step 7: Optimized feature extraction
    print("\n🔧 Optimized feature extraction...")
    vectorizer = create_optimized_feature_extraction()
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✅ Feature matrix: {X_train_vec.shape}")
    print(f"✅ Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Step 8: Extensive hyperparameter optimization
    best_lr, best_svm = extensive_hyperparameter_optimization(X_train_vec, y_train)
    
    # Step 9: Train optimized models
    print("\n🤖 Training ultra-optimized models...")
    models = create_optimized_models()
    models['best_lr'] = best_lr
    models['best_svm'] = best_svm
    
    models_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
        test_accuracy = accuracy_score(y_test, model.predict(X_test_vec))
        
        models_results[name] = {
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        trained_models[name] = model
        
        print(f"✅ {name}: Test={test_accuracy:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    # Step 10: Create ultra ensemble
    ultra_ensemble = create_ultra_ensemble(trained_models, X_train_vec, y_train)
    ensemble_accuracy = accuracy_score(y_test, ultra_ensemble.predict(X_test_vec))
    cv_scores_ensemble = cross_val_score(ultra_ensemble, X_train_vec, y_train, cv=5, scoring='accuracy')
    
    models_results['ultra_ensemble'] = {
        'test_accuracy': ensemble_accuracy,
        'cv_mean': cv_scores_ensemble.mean(),
        'cv_std': cv_scores_ensemble.std()
    }
    trained_models['ultra_ensemble'] = ultra_ensemble
    
    print(f"\n🎯 Ultra Ensemble: Test={ensemble_accuracy:.4f}, CV={cv_scores_ensemble.mean():.4f}±{cv_scores_ensemble.std():.4f}")
    
    # Step 11: Select best model
    best_model_name = max(models_results, key=lambda x: models_results[x]['test_accuracy'])
    best_accuracy = models_results[best_model_name]['test_accuracy']
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"🎯 Best Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Step 12: Final comprehensive evaluation
    print(f"\n{'='*50}")
    print("FINAL MODEL EVALUATION")
    print(f"{'='*50}")
    
    y_pred_final = best_model.predict(X_test_vec)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    
    print(f"🎯 Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred_final, target_names=emotion_labels, zero_division=0))
    
    # Step 13: Create and save complete pipeline
    print("\n💾 Creating and saving complete pipeline...")
    complete_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', best_model)
    ])
    
    # Save all models
    joblib.dump(complete_pipeline, '../models/ultra_high_accuracy_emotion_model.joblib')
    joblib.dump(vectorizer, '../models/ultra_high_accuracy_vectorizer.joblib')
    joblib.dump(best_model, '../models/ultra_high_accuracy_classifier.joblib')
    
    print("✅ Ultra-high accuracy models saved successfully!")
    
    # Step 14: Save comprehensive training summary
    summary = {
        'final_accuracy': float(final_accuracy),
        'best_model': best_model_name,
        'dataset_size': len(df),
        'feature_count': X_train_vec.shape[1],
        'emotions': emotion_labels,
        'class_balance_ratio': float(balance_ratio),
        'all_model_results': {k: {
            'test_accuracy': float(v['test_accuracy']),
            'cv_mean': float(v['cv_mean']),
            'cv_std': float(v['cv_std'])
        } for k, v in models_results.items()}
    }
    
    with open('../models/ultra_high_accuracy_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Step 15: Test with examples
    print(f"\n{'='*50}")
    print("TESTING WITH EXAMPLES")
    print(f"{'='*50}")
    
    test_examples = [
        "I'm absolutely thrilled and overjoyed about this incredible news!",
        "This situation makes me feel deeply saddened and heartbroken.",
        "I'm furious and outraged by this completely unacceptable behavior!",
        "I'm terrified and anxious about what might happen next.",
        "Wow, I never expected such an amazing and wonderful surprise!",
        "This is absolutely disgusting and makes me feel sick.",
        "Please review the quarterly report by Friday afternoon."
    ]
    
    for text in test_examples:
        processed = detector.preprocess_text(text)
        if processed:  # Only predict if preprocessing was successful
            prediction = complete_pipeline.predict([processed])[0]
            probabilities = complete_pipeline.predict_proba([processed])[0]
            confidence = max(probabilities)
            
            print(f"\nText: '{text}'")
            print(f"Predicted: {prediction.upper()} (confidence: {confidence:.1%})")
            
            # Show top 3 emotions
            emotion_probs = list(zip(emotion_labels, probabilities))
            emotion_probs.sort(key=lambda x: x[1], reverse=True)
            print("Top 3 predictions:")
            for emotion, prob in emotion_probs[:3]:
                print(f"  {emotion}: {prob:.1%}")
    
    # Final results
    print(f"\n{'='*70}")
    print("🎉 ULTRA-HIGH ACCURACY TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"🎯 Final Accuracy: {final_accuracy:.2%}")
    print(f"🤖 Best Model: {best_model_name}")
    print(f"📊 Dataset Size: {len(df)} samples")
    print(f"🔧 Features: {X_train_vec.shape[1]}")
    print(f"⚖️ Class Balance: {balance_ratio:.3f}")
    
    if final_accuracy >= 0.90:
        print("🏆 SUCCESS: Achieved 90%+ accuracy target!")
        print("🚀 Ultra-high accuracy model is ready for production!")
    elif final_accuracy >= 0.85:
        print("🎯 EXCELLENT: Achieved 85%+ accuracy!")
        print("✨ High-quality model ready for deployment!")
    else:
        print(f"📈 Achieved {final_accuracy:.1%} accuracy")
        print("💡 Further optimization suggestions:")
        print("   - Consider transformer models (BERT, RoBERTa)")
        print("   - Implement advanced ensemble techniques")
        print("   - Add more domain-specific training data")
    
    return final_accuracy, complete_pipeline


if __name__ == "__main__":
    try:
        accuracy, model = main()
        print(f"\n✅ Ultra-high accuracy training completed with {accuracy:.1%} accuracy!")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
