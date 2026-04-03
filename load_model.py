import joblib

# Define the path to your model file
model_path = "models/emotion_model_comprehensive_improved.joblib"

try:
    # Load the model from the file
    loaded_model = joblib.load(model_path)

    # Print the loaded model object to inspect it
    print("✅ Model loaded successfully!")
    print("\nModel details:")
    print(loaded_model)

except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at '{model_path}'")
except Exception as e:
    print(f"❌ An error occurred: {e}")

