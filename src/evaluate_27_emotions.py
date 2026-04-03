"""
Evaluate accuracy for all 27 emotions and overall model performance
"""
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

def load_and_evaluate_models():
    """Load models and evaluate performance for all 27 emotions"""
    
    print("🔍 COMPREHENSIVE EMOTION DETECTION ACCURACY EVALUATION")
    print("=" * 80)
    
    # Load the existing classification report for 27 emotions
    try:
        with open('../models/1350k_classification_report.json', 'r') as f:
            classification_data = json.load(f)
        
        print("📊 CURRENT MODEL PERFORMANCE (27 Emotions)")
        print("-" * 80)
        
        # Extract overall accuracy
        overall_accuracy = classification_data.get('accuracy', 0)
        print(f"🎯 OVERALL MODEL ACCURACY: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print()
        
        # Create a list to store individual emotion accuracies
        emotion_accuracies = []
        emotions_below_95 = []
        emotions_above_95 = []
        
        print("📋 INDIVIDUAL EMOTION ACCURACIES:")
        print("-" * 80)
        
        # Get all emotions (exclude summary statistics)
        exclude_keys = ['accuracy', 'macro avg', 'weighted avg']
        emotions = [key for key in classification_data.keys() if key not in exclude_keys]
        
        for emotion in sorted(emotions):
            emotion_data = classification_data[emotion]
            
            # Use F1-score as the primary accuracy metric for individual emotions
            # as it balances precision and recall
            f1_score = emotion_data.get('f1-score', 0)
            precision = emotion_data.get('precision', 0)
            recall = emotion_data.get('recall', 0)
            support = emotion_data.get('support', 0)
            
            emotion_accuracies.append(f1_score)
            
            # Categorize emotions based on performance
            if f1_score >= 0.95:
                emotions_above_95.append((emotion, f1_score))
                status = "🟢 EXCELLENT"
            elif f1_score >= 0.90:
                status = "🟡 VERY GOOD"
            elif f1_score >= 0.80:
                status = "🟠 GOOD"
            elif f1_score >= 0.70:
                status = "🔴 NEEDS WORK"
            else:
                status = "❌ POOR"
                emotions_below_95.append((emotion, f1_score))
            
            print(f"{emotion.replace('_', ' ').title():25} | "
                  f"F1: {f1_score:.3f} ({f1_score*100:5.1f}%) | "
                  f"Precision: {precision:.3f} | "
                  f"Recall: {recall:.3f} | "
                  f"Support: {int(support):4d} | {status}")
        
        print()
        print("=" * 80)
        print("📈 PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Calculate statistics
        avg_emotion_accuracy = np.mean(emotion_accuracies)
        min_emotion_accuracy = np.min(emotion_accuracies)
        max_emotion_accuracy = np.max(emotion_accuracies)
        
        print(f"📊 Total Emotions Evaluated: {len(emotions)}")
        print(f"🎯 Overall Model Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"📈 Average Emotion F1-Score: {avg_emotion_accuracy:.4f} ({avg_emotion_accuracy*100:.2f}%)")
        print(f"📉 Lowest Emotion F1-Score: {min_emotion_accuracy:.4f} ({min_emotion_accuracy*100:.2f}%)")
        print(f"📊 Highest Emotion F1-Score: {max_emotion_accuracy:.4f} ({max_emotion_accuracy*100:.2f}%)")
        print()
        
        # Performance categories
        excellent_count = len([acc for acc in emotion_accuracies if acc >= 0.95])
        very_good_count = len([acc for acc in emotion_accuracies if 0.90 <= acc < 0.95])
        good_count = len([acc for acc in emotion_accuracies if 0.80 <= acc < 0.90])
        needs_work_count = len([acc for acc in emotion_accuracies if 0.70 <= acc < 0.80])
        poor_count = len([acc for acc in emotion_accuracies if acc < 0.70])
        
        print("🏆 PERFORMANCE DISTRIBUTION:")
        print(f"🟢 Excellent (≥95%):     {excellent_count:2d} emotions ({excellent_count/len(emotions)*100:.1f}%)")
        print(f"🟡 Very Good (90-95%):   {very_good_count:2d} emotions ({very_good_count/len(emotions)*100:.1f}%)")
        print(f"🟠 Good (80-90%):        {good_count:2d} emotions ({good_count/len(emotions)*100:.1f}%)")
        print(f"🔴 Needs Work (70-80%):  {needs_work_count:2d} emotions ({needs_work_count/len(emotions)*100:.1f}%)")
        print(f"❌ Poor (<70%):          {poor_count:2d} emotions ({poor_count/len(emotions)*100:.1f}%)")
        print()
        
        if emotions_above_95:
            print("🎉 EMOTIONS WITH 95%+ ACCURACY:")
            for emotion, score in sorted(emotions_above_95, key=lambda x: x[1], reverse=True):
                print(f"   {emotion.replace('_', ' ').title():25} | {score:.3f} ({score*100:.1f}%)")
            print()
        
        if emotions_below_95:
            print("⚠️  EMOTIONS NEEDING IMPROVEMENT (Below 95%):")
            for emotion, score in sorted(emotions_below_95, key=lambda x: x[1]):
                print(f"   {emotion.replace('_', ' ').title():25} | {score:.3f} ({score*100:.1f}%)")
            print()
        
        # Goal assessment
        print("🎯 ACCURACY GOAL ASSESSMENT:")
        print("-" * 80)
        
        if excellent_count == len(emotions):
            print("🏆 SUCCESS! All emotions have achieved 95%+ accuracy!")
        else:
            remaining = len(emotions) - excellent_count
            print(f"📈 Progress: {excellent_count}/{len(emotions)} emotions have 95%+ accuracy")
            print(f"🔧 Need improvement: {remaining} emotions require optimization")
            
        print()
        
        # Load comprehensive test results if available
        try:
            with open('../comprehensive_test_results.json', 'r') as f:
                comprehensive_results = json.load(f)
            
            print("🧪 COMPREHENSIVE TEST RESULTS (5000 samples per emotion):")
            print("-" * 80)
            
            comprehensive_emotions_95_plus = []
            comprehensive_emotions_below_95 = []
            
            for emotion, data in comprehensive_results.items():
                accuracy = data.get('accuracy', 0) / 100  # Convert percentage to decimal
                status = data.get('status', 'UNKNOWN')
                samples = data.get('samples_tested', 0)
                correct = data.get('correct_predictions', 0)
                
                if accuracy >= 0.95:
                    comprehensive_emotions_95_plus.append((emotion, accuracy))
                    status_icon = "🟢"
                else:
                    comprehensive_emotions_below_95.append((emotion, accuracy))
                    status_icon = "🔴" if accuracy < 0.80 else "🟡"
                
                print(f"{emotion.replace('_', ' ').title():25} | "
                      f"{accuracy:.3f} ({accuracy*100:5.1f}%) | "
                      f"{correct:4d}/{samples} | {status_icon} {status}")
            
            print()
            print(f"🎯 Comprehensive Test Summary:")
            print(f"   Emotions with 95%+ accuracy: {len(comprehensive_emotions_95_plus)}/{len(comprehensive_results)}")
            print(f"   Average accuracy: {np.mean([data['accuracy']/100 for data in comprehensive_results.values()]):.3f}")
            
        except FileNotFoundError:
            print("⚠️  Comprehensive test results not found. Run comprehensive testing for detailed per-emotion accuracy.")
        
        return {
            'overall_accuracy': overall_accuracy,
            'average_emotion_accuracy': avg_emotion_accuracy,
            'emotions_above_95': len(emotions_above_95),
            'emotions_below_95': len(emotions_below_95),
            'total_emotions': len(emotions),
            'emotion_accuracies': dict(zip(emotions, emotion_accuracies))
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find classification report file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading model data: {e}")
        return None

def evaluate_ultra_high_accuracy_model():
    """Evaluate the newly trained ultra-high accuracy model"""
    
    print("\n🚀 EVALUATING ULTRA-HIGH ACCURACY MODEL")
    print("=" * 80)
    
    try:
        # Load the ultra-high accuracy model
        model_path = '../models/ultra_high_accuracy_emotion_model.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Successfully loaded ultra-high accuracy model")
            
            # Load training summary
            summary_path = '../models/ultra_high_accuracy_training_summary.json'
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                print(f"📊 Ultra-High Accuracy Model Performance:")
                print(f"   Final Accuracy: {summary['final_accuracy']:.4f} ({summary['final_accuracy']*100:.2f}%)")
                print(f"   Best Model: {summary['best_model']}")
                print(f"   Dataset Size: {summary['dataset_size']}")
                print(f"   Feature Count: {summary['feature_count']}")
                print(f"   Emotions Covered: {len(summary['emotions'])}")
                print(f"   Emotions: {', '.join(summary['emotions'])}")
                
                print(f"\n📈 Individual Model Results:")
                for model_name, results in summary['all_model_results'].items():
                    print(f"   {model_name:30} | Test: {results['test_accuracy']:.4f} | CV: {results['cv_mean']:.4f}±{results['cv_std']:.4f}")
            
        else:
            print(f"⚠️  Ultra-high accuracy model not found at {model_path}")
            
    except Exception as e:
        print(f"❌ Error evaluating ultra-high accuracy model: {e}")

if __name__ == "__main__":
    # Evaluate both models
    results = load_and_evaluate_models()
    evaluate_ultra_high_accuracy_model()
    
    if results:
        print("\n" + "=" * 80)
        print("🎯 FINAL SUMMARY")
        print("=" * 80)
        print(f"Overall Model Accuracy: {results['overall_accuracy']*100:.2f}%")
        print(f"Average Individual Emotion Accuracy: {results['average_emotion_accuracy']*100:.2f}%")
        print(f"Emotions at 95%+ accuracy: {results['emotions_above_95']}/{results['total_emotions']}")
        
        if results['emotions_above_95'] == results['total_emotions']:
            print("🏆 MISSION ACCOMPLISHED: All emotions have 95%+ accuracy!")
        else:
            remaining = results['total_emotions'] - results['emotions_above_95']
            print(f"🔧 Work remaining: {remaining} emotions need optimization to reach 95%+")
