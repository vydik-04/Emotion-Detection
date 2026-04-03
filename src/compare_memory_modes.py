"""
Memory Optimization Mode Comparison Script

This script runs all three memory optimization modes and compares their:
- Accuracy
- Training time
- Memory usage
- Model size
"""

import time
import os
import json
from memory_optimized_emotion_detector import MemoryOptimizedEmotionModel
import pandas as pd

def run_memory_mode_comparison():
    """Run all three memory modes and compare results."""
    
    print("🔍 COMPREHENSIVE MEMORY MODE COMPARISON")
    print("=" * 80)
    
    dataset_path = '../data/emotion_dataset_1350k.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    modes = ['aggressive', 'balanced', 'conservative']
    results = {}
    
    for mode in modes:
        print(f"\n🚀 Testing {mode.upper()} Mode...")
        print("-" * 60)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Initialize model
            model = MemoryOptimizedEmotionModel(memory_mode=mode)
            
            # Train model
            training_results = model.train_memory_efficient_model(dataset_path)
            
            # Record end time
            end_time = time.time()
            training_time = end_time - start_time
            
            # Get model size
            model_path = training_results['model_path']
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Store results
            results[mode] = {
                'test_accuracy': training_results['test_accuracy'],
                'cv_accuracy': training_results['cv_accuracy'],
                'training_time_seconds': training_time,
                'model_size_mb': model_size_mb,
                'config': model.config,
                'emotions_count': len(model.label_encoder.classes_)
            }
            
            print(f"✅ {mode.upper()} completed in {training_time:.1f} seconds")
            
        except Exception as e:
            print(f"❌ {mode.upper()} failed: {e}")
            results[mode] = {'error': str(e)}
    
    # Display comparison results
    display_comparison_results(results)
    
    # Save comparison results
    save_comparison_results(results)
    
    return results

def display_comparison_results(results):
    """Display a formatted comparison of all results."""
    
    print("\n" + "=" * 80)
    print("📊 MEMORY OPTIMIZATION COMPARISON RESULTS")
    print("=" * 80)
    
    # Create comparison table
    print(f"{'Mode':<12} {'Accuracy':<10} {'CV Score':<10} {'Time(s)':<10} {'Size(MB)':<10} {'Features':<10} {'Dataset%':<10}")
    print("-" * 80)
    
    for mode, data in results.items():
        if 'error' not in data:
            accuracy = f"{data['test_accuracy']*100:.1f}%"
            cv_score = f"{data['cv_accuracy']*100:.1f}%"
            time_str = f"{data['training_time_seconds']:.1f}"
            size_str = f"{data['model_size_mb']:.1f}"
            features = f"{data['config']['max_features']}"
            dataset_pct = f"{data['config']['dataset_fraction']*100:.1f}%"
            
            print(f"{mode:<12} {accuracy:<10} {cv_score:<10} {time_str:<10} {size_str:<10} {features:<10} {dataset_pct:<10}")
        else:
            print(f"{mode:<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
    
    print("\n📈 Performance Analysis:")
    
    # Find best performing mode
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_accuracy = max(valid_results.items(), key=lambda x: x[1]['test_accuracy'])
        fastest_training = min(valid_results.items(), key=lambda x: x[1]['training_time_seconds'])
        smallest_model = min(valid_results.items(), key=lambda x: x[1]['model_size_mb'])
        
        print(f"🏆 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['test_accuracy']*100:.2f}%)")
        print(f"⚡ Fastest Training: {fastest_training[0]} ({fastest_training[1]['training_time_seconds']:.1f}s)")
        print(f"💾 Smallest Model: {smallest_model[0]} ({smallest_model[1]['model_size_mb']:.1f}MB)")
        
        # Memory vs Accuracy trade-off analysis
        print("\n🔍 Trade-off Analysis:")
        for mode, data in valid_results.items():
            memory_efficiency = data['config']['max_features'] * data['config']['dataset_fraction']
            accuracy_per_memory = data['test_accuracy'] / memory_efficiency if memory_efficiency > 0 else 0
            
            print(f"  {mode:12}: Accuracy/Memory ratio = {accuracy_per_memory*1000:.3f}")

def save_comparison_results(results):
    """Save comparison results to a JSON file."""
    
    output_dir = '../models'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'memory_mode_comparison.json')
    
    # Add timestamp
    results['comparison_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comparison results saved to: {output_path}")

def test_individual_mode(mode='balanced'):
    """Test a specific memory mode."""
    
    print(f"🧪 Testing {mode.upper()} Mode Individually")
    print("-" * 50)
    
    dataset_path = '../data/emotion_dataset_1350k.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return None
    
    try:
        # Initialize and train model
        model = MemoryOptimizedEmotionModel(memory_mode=mode)
        results = model.train_memory_efficient_model(dataset_path)
        
        # Test some predictions
        test_predictions(model)
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing {mode} mode: {e}")
        return None

def test_predictions(model):
    """Test the model with sample predictions."""
    
    print("\n🧪 Testing Sample Predictions:")
    print("-" * 40)
    
    test_texts = [
        "I am absolutely thrilled and excited about this opportunity!",
        "This situation makes me feel very sad and disappointed.",
        "I'm really angry about what happened today.",
        "That movie was absolutely terrifying and scary.",
        "What a beautiful and amazing surprise!",
        "This food tastes absolutely disgusting.",
        "I feel calm and peaceful right now.",
        "This is so confusing and I don't understand.",
        "I'm feeling quite anxious about the presentation.",
        "The sunset is so beautiful, it fills me with awe."
    ]
    
    for i, text in enumerate(test_texts, 1):
        try:
            prediction = model.predict_memory_efficient(text)
            emotion = prediction['emotion']
            confidence = prediction['confidence']
            
            print(f"{i:2d}. Text: {text[:50]}...")
            print(f"    Emotion: {emotion} (confidence: {confidence:.3f})")
            print()
            
        except Exception as e:
            print(f"{i:2d}. Error predicting: {e}")

def create_custom_config():
    """Create a custom memory configuration."""
    
    print("🔧 Custom Memory Configuration Creator")
    print("-" * 50)
    
    try:
        max_features = int(input("Max TF-IDF features (100-10000): ") or "2000")
        dataset_fraction = float(input("Dataset fraction (0.01-0.5): ") or "0.08")
        enable_smote = input("Enable SMOTE? (y/n): ").lower().startswith('y')
        n_estimators = int(input("Number of estimators (25-200): ") or "75")
        
        custom_config = {
            'max_features': max_features,
            'dataset_fraction': dataset_fraction,
            'ngram_range': (1, 2),
            'batch_size': 2000,
            'cv_folds': 3,
            'enable_smote': enable_smote,
            'n_estimators': n_estimators
        }
        
        print(f"\n✅ Custom configuration created: {custom_config}")
        
        # Test custom configuration
        print("\n🧪 Testing custom configuration...")
        
        # Create a modified model class for custom config
        class CustomMemoryModel(MemoryOptimizedEmotionModel):
            def _get_memory_config(self):
                return custom_config
        
        model = CustomMemoryModel(memory_mode='custom')
        
        dataset_path = '../data/emotion_dataset_1350k.csv'
        if os.path.exists(dataset_path):
            results = model.train_memory_efficient_model(dataset_path)
            print(f"\n🎯 Custom model accuracy: {results['test_accuracy']*100:.2f}%")
        else:
            print("❌ Dataset not found for testing")
            
    except Exception as e:
        print(f"❌ Error creating custom config: {e}")

def main():
    """Main function with interactive options."""
    
    print("🚀 MEMORY OPTIMIZATION TESTING SUITE")
    print("=" * 80)
    
    while True:
        print("\n🔧 Available Options:")
        print("1. Run full comparison (all 3 modes)")
        print("2. Test specific mode")
        print("3. Create custom configuration")
        print("4. View previous results")
        print("5. Exit")
        
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                run_memory_mode_comparison()
                
            elif choice == '2':
                print("\nAvailable modes:")
                print("1. Aggressive (minimal memory)")
                print("2. Balanced (good balance)")
                print("3. Conservative (better accuracy)")
                
                mode_choice = input("Select mode (1-3): ").strip()
                mode_map = {'1': 'aggressive', '2': 'balanced', '3': 'conservative'}
                mode = mode_map.get(mode_choice, 'balanced')
                
                test_individual_mode(mode)
                
            elif choice == '3':
                create_custom_config()
                
            elif choice == '4':
                results_path = '../models/memory_mode_comparison.json'
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    print("\n📊 Previous Results:")
                    print(json.dumps(results, indent=2))
                else:
                    print("❌ No previous results found")
                    
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
