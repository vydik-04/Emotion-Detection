import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_emotion_model_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Parallel Multi-Classifier System for Emotion Detection', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Define colors
    input_color = '#E8F4FD'
    process_color = '#D6EAF8'  
    model_color = '#D5F5E3'
    logic_color = '#FAD7A0'
    output_color = '#FDEDEC'
    
    # Input Layer
    input_box = FancyBboxPatch((4, 10), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=input_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(5, 10.4, 'Input Text\n(Raw Text Data)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ML Processing Path (Left Side)
    # Step 1: Preprocessing
    preprocess_box = FancyBboxPatch((0.5, 8.5), 3, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=process_color, edgecolor='black', linewidth=1)
    ax.add_patch(preprocess_box)
    ax.text(2, 8.9, 'Negation-Aware Preprocessing\n(Tokenize, Lemmatize, Clean)', 
            ha='center', va='center', fontsize=9)
    
    # Step 2: Vectorization
    vector_box = FancyBboxPatch((0.5, 7), 3, 0.8, boxstyle="round,pad=0.1",
                               facecolor=process_color, edgecolor='black', linewidth=1)
    ax.add_patch(vector_box)
    ax.text(2, 7.4, 'TF-IDF Vectorization\n(Text to Numerical Features)', 
            ha='center', va='center', fontsize=9)
    
    # Step 3: ML Model
    ml_model_box = FancyBboxPatch((0.5, 5.5), 3, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=model_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(ml_model_box)
    ax.text(2, 5.9, 'ML Ensemble Model\n(Logistic Regression + SVM)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # ML Output
    ml_output_box = FancyBboxPatch((0.5, 4), 3, 0.8, boxstyle="round,pad=0.1",
                                  facecolor=input_color, edgecolor='black', linewidth=1)
    ax.add_patch(ml_output_box)
    ax.text(2, 4.4, 'ML Predictions\n& Probabilities', ha='center', va='center', fontsize=9)
    
    # Rule-Based Path (Right Side)
    rule_box = FancyBboxPatch((6.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                             facecolor=process_color, edgecolor='black', linewidth=1)
    ax.add_patch(rule_box)
    ax.text(8, 8.25, 'Rule-Based Analysis\n• 27 Emotion Keywords\n• Pattern Matching\n• Context Detection', 
            ha='center', va='center', fontsize=9)
    
    # Hybrid Decision Logic (Diamond shape)
    logic_points = np.array([[5, 3], [6.5, 2.25], [5, 1.5], [3.5, 2.25], [5, 3]])
    logic_diamond = patches.Polygon(logic_points, closed=True, 
                                   facecolor=logic_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(logic_diamond)
    ax.text(5, 2.25, 'Hybrid\nDecision\nLogic', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Final Output
    output_ellipse = patches.Ellipse((5, 0.5), 3.5, 0.8, 
                                   facecolor=output_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(output_ellipse)
    ax.text(5, 0.5, 'Final Emotion Classification\n(Top 3 emotions with confidence scores)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows for ML Path
    # Input to Preprocessing
    ax.annotate('', xy=(2, 9.3), xytext=(4.5, 10.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Preprocessing to Vectorization
    ax.annotate('', xy=(2, 7.8), xytext=(2, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Vectorization to ML Model
    ax.annotate('', xy=(2, 6.3), xytext=(2, 7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # ML Model to ML Output
    ax.annotate('', xy=(2, 4.8), xytext=(2, 5.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Add arrow for Rule-Based Path
    # Input to Rules
    ax.annotate('', xy=(7, 8.8), xytext=(5.5, 10.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # ML Output to Hybrid Logic
    ax.annotate('', xy=(3.8, 2.8), xytext=(2.8, 4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    ax.text(2.2, 3.2, 'ML Results', fontsize=8, color='blue', rotation=35)
    
    # Rules to Hybrid Logic
    ax.annotate('', xy=(6.2, 2.8), xytext=(7.5, 7.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
    ax.text(7.5, 5, 'Rule\nMatches', fontsize=8, color='green', rotation=-55)
    
    # Hybrid Logic to Output
    ax.annotate('', xy=(5, 1.3), xytext=(5, 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(5.5, 1.2, 'Final Decision', fontsize=8, color='red')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=input_color, label='Input/Output'),
        patches.Patch(color=process_color, label='Processing Steps'),
        patches.Patch(color=model_color, label='ML Models'),
        patches.Patch(color=logic_color, label='Decision Logic'),
        patches.Patch(color=output_color, label='Final Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add accuracy info
    ax.text(0.5, 0.5, '🎯 System Accuracy: 99%+\n🧠 27 Emotions Detected\n⚡ Real-time Processing', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('emotion_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Emotion Detection Model Architecture diagram created successfully!")
    print("📁 Saved as: emotion_model_architecture.png")

if __name__ == "__main__":
    create_emotion_model_architecture()
