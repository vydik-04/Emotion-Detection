#!/usr/bin/env python3
"""
Inspects the main .joblib model file to display its contents.
"""

import joblib
import os

def inspect_model(model_path):
    """Loads and inspects a joblib model file."""
    print(f"🔍 Inspecting model file: {model_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at '{model_path}'")
        return
        
    try:
        # Load the model from the file
        model = joblib.load(model_path)
        
        print("✅ Model loaded successfully!\n")
        
        print("MODEL DETAILS")
        print("-" * 20)
        
        # Print the model object and its type
        print(f"🔹 Model Object: {model}\n")
        print(f"🔹 Model Type: {type(model)}\n")
        
        # If it's a scikit-learn pipeline, inspect its steps
        if hasattr(model, 'steps'):
            print("🔹 Pipeline Steps:")
            for i, (name, step) in enumerate(model.steps, 1):
                step_details = str(step).replace('\n', '\n' + ' ' * 8)
                print(f"  {i}. {name}")
                print(f"     └─ {step_details}")
            print()
        
        # If it has parameters, show them
        if hasattr(model, 'get_params'):
            print("🔹 Model Parameters:")
            params = model.get_params(deep=False) # Get top-level params
            for param, value in params.items():
                value_str = str(value)
                if len(value_str) > 120:
                    value_str = value_str[:120] + "..."
                print(f"  - {param}: {value_str}")

    except Exception as e:
        print(f"❌ An error occurred while loading or inspecting the model: {e}")

if __name__ == "__main__":
    main_model_path = "models/emotion_model_comprehensive_improved.joblib"
    inspect_model(main_model_path)

