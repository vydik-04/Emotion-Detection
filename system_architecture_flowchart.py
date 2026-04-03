"""
System Architecture Flowchart Generator
Creates a flowchart-style diagram inspired by the user's reference image.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_system_architecture_flowchart():
    """Create a flowchart-style system architecture diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Common styles for boxes and arrows
    box_style = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black', linewidth=2)
    arrow_style = dict(arrowstyle="->", connectionstyle="arc3", color="black", lw=2)

    # 1. User Input
    ax.text(6, 13, "User Text Input\n(Streamlit UI)", ha='center', va='center', bbox=box_style, size=10)

    # 2. Pre-processing
    ax.text(6, 11, "Text Pre-processing\n(Lowercase, Cleaning, Lemmatization)", ha='center', va='center', bbox=box_style, size=10)

    # Arrow from Input to Pre-processing
    ax.annotate("", xy=(6, 11.5), xytext=(6, 12.5), arrowprops=arrow_style)

    # Arrow forking from Pre-processing
    ax.annotate("Cleaned Text", xy=(3.5, 9.5), xytext=(6, 10.5), arrowprops=arrow_style, ha='center', va='center', size=9)
    ax.annotate("Cleaned Text", xy=(8.5, 9.5), xytext=(6, 10.5), arrowprops=arrow_style, ha='center', va='center', size=9)

    # --- Main ML Pipeline (Left) ---
    # Red container box
    red_box = patches.FancyBboxPatch((0.5, 3.5), 5, 5, boxstyle="round,pad=0.5", ec='red', fc='none', lw=2)
    ax.add_patch(red_box)

    # 3a. Vectorizer
    ax.text(3, 8.5, "TF-IDF Vectorizer", ha='center', va='center', bbox=box_style, size=10)
    ax.annotate("", xy=(3, 9), xytext=(3.5, 9.5), arrowprops=arrow_style)
    ax.annotate("Feature Vectors", xy=(3, 7.5), xytext=(3, 8), arrowprops=arrow_style, ha='center', size=9)

    # 4a. ML Model
    ax.text(3, 6.5, "ML Model\n(Logistic Regression / SVM)", ha='center', va='center', bbox=box_style, size=10)
    ax.annotate("", xy=(3, 7), xytext=(3, 6), arrowprops=dict(arrowstyle="-", color="black", lw=2))

    # 5a. Output Layer
    ax.text(3, 4.5, "Probability & Prediction\nLayer (Softmax)", ha='center', va='center', bbox=box_style, size=10)
    ax.annotate("", xy=(3, 5), xytext=(3, 6), arrowprops=arrow_style)

    # --- Rule-Based Pipeline (Right) ---
    # 3b. Rule Engine
    ax.text(9, 8.5, "Rule-Based Engine", ha='center', va='center', bbox=box_style, size=10)
    ax.annotate("", xy=(9, 9), xytext=(8.5, 9.5), arrowprops=arrow_style)

    # 4b. Keyword Analysis
    ax.text(9, 6.5, "Keyword & Context\nAnalysis", ha='center', va='center', bbox=box_style, size=10)
    ax.annotate("", xy=(9, 7), xytext=(9, 8), arrowprops=arrow_style)
    
    # --- Combination and Final Output ---
    # 6. Hybrid Analysis
    ax.text(6, 2, "Hybrid Analysis Engine\n(ML + Rules)", ha='center', va='center', bbox=box_style, size=10)
    
    # Arrow from ML pipeline to Hybrid Analysis
    ax.annotate("ML Emotion Prediction", xy=(5.5, 2.5), xytext=(3, 4), arrowprops=arrow_style, ha='right', size=9)
    
    # Arrow from Rule pipeline to Hybrid Analysis
    ax.annotate("Rule-Based Override", xy=(6.5, 2.5), xytext=(9, 6), arrowprops=arrow_style, ha='left', size=9)

    # 7. Final Output
    ax.text(6, 0.5, "Final Output Result\n(Displayed in UI)", ha='center', va='center', bbox=box_style, size=10)
    ax.annotate("Final Emotion Score", xy=(6, 1), xytext=(6, 1.5), arrowprops=arrow_style, ha='center', size=9)
    
    # Feedback Loop
    feedback_arrow = patches.ConnectionPatch((4.5, 0.5), (0, 7), "data", "data", 
                                             arrowstyle="->", connectionstyle="arc3,rad=-0.4", 
                                             color="black", lw=2)
    ax.add_patch(feedback_arrow)
    ax.text(0, 7.5, "User Feedback &\nRetraining Cycle", size=9, ha='center')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 Generating System Architecture Flowchart...")
    fig = create_system_architecture_flowchart()
    output_path = "System_Architecture_Flowchart.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Flowchart saved as: {output_path}")
