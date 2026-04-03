"""
Implementation Tools Slide Generator
Creates a professional slide listing the project's implementation tools.
"""

import matplotlib.pyplot as plt

def create_implementation_tools_slide():
    """Create a dark-themed slide listing implementation tools."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.patch.set_facecolor('#0D1B2A') # Dark Navy Blue Background
    ax.axis('off')

    # Title
    ax.text(8, 8, 'Implementation Tools', ha='center', va='center', 
            fontsize=36, fontweight='bold', color='#E0E1DD')

    # Define colors and styles
    cat_color = '#778DA9'  # Lighter blue for category titles
    item_color = '#E0E1DD' # Off-white for items
    
    # --- Column 1 ---
    x1 = 2
    y_start1 = 6.5
    
    # Core & Data Science
    ax.text(x1, y_start1, 'Core & Data Science', ha='left', va='center', 
            fontsize=18, fontweight='bold', color=cat_color)
    ax.plot([x1, x1 + 4], [y_start1 - 0.3, y_start1 - 0.3], color=cat_color, lw=1)
    ax.text(x1, y_start1 - 0.7, '• Python 3.10', fontsize=14, color=item_color)
    ax.text(x1, y_start1 - 1.2, '• Scikit-Learn', fontsize=14, color=item_color)
    ax.text(x1, y_start1 - 1.7, '• Pandas', fontsize=14, color=item_color)
    ax.text(x1, y_start1 - 2.2, '• NumPy', fontsize=14, color=item_color)

    # NLP Libraries
    y_start2 = 3.5
    ax.text(x1, y_start2, 'Natural Language Processing', ha='left', va='center', 
            fontsize=18, fontweight='bold', color=cat_color)
    ax.plot([x1, x1 + 4], [y_start2 - 0.3, y_start2 - 0.3], color=cat_color, lw=1)
    ax.text(x1, y_start2 - 0.7, '• NLTK (Natural Language Toolkit)', fontsize=14, color=item_color)
    ax.text(x1, y_start2 - 1.2, '• SpaCy', fontsize=14, color=item_color)
    ax.text(x1, y_start2 - 1.7, '• TextBlob', fontsize=14, color=item_color)
    ax.text(x1, y_start2 - 2.2, '• Transformers (Hugging Face)', fontsize=14, color=item_color)

    # --- Column 2 ---
    x2 = 10
    y_start3 = 6.5

    # Web & Visualization
    ax.text(x2, y_start3, 'Web & Visualization', ha='left', va='center', 
            fontsize=18, fontweight='bold', color=cat_color)
    ax.plot([x2, x2 + 4], [y_start3 - 0.3, y_start3 - 0.3], color=cat_color, lw=1)
    ax.text(x2, y_start3 - 0.7, '• Streamlit', fontsize=14, color=item_color)
    ax.text(x2, y_start3 - 1.2, '• Flask', fontsize=14, color=item_color)
    ax.text(x2, y_start3 - 1.7, '• Plotly', fontsize=14, color=item_color)
    ax.text(x2, y_start3 - 2.2, '• Matplotlib & Seaborn', fontsize=14, color=item_color)
    
    # Development & Deployment
    y_start4 = 3.5
    ax.text(x2, y_start4, 'Development & Deployment', ha='left', va='center', 
            fontsize=18, fontweight='bold', color=cat_color)
    ax.plot([x2, x2 + 4], [y_start4 - 0.3, y_start4 - 0.3], color=cat_color, lw=1)
    ax.text(x2, y_start4 - 0.7, '• Jupyter Notebooks', fontsize=14, color=item_color)
    ax.text(x2, y_start4 - 1.2, '• Joblib (Model Serialization)', fontsize=14, color=item_color)
    ax.text(x2, y_start4 - 1.7, '• Pytest (Testing)', fontsize=14, color=item_color)
    ax.text(x2, y_start4 - 2.2, '• Git & GitHub (Version Control)', fontsize=14, color=item_color)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 Generating Implementation Tools Slide...")
    fig = create_implementation_tools_slide()
    output_path = "Implementation_Tools_Slide.png"
    fig.savefig(output_path, dpi=300, facecolor='#0D1B2A')
    print(f"✅ Slide saved as: {output_path}")
