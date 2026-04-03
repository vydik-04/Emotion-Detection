"""
Professional Module Design Diagram Generator
Creates a clean, slide-style module design diagram matching the reference format
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_professional_module_design():
    """Create a professional, clean module design diagram."""
    
    # Set up the figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Title
    ax.text(8, 9.5, 'MODULE DESIGN', 
            fontsize=28, fontweight='bold', ha='center', color='#2C3E50')
    
    # Define module positions (2x3 grid)
    positions = [
        (1, 6.5),   # Module 1 - Top Left
        (8, 6.5),   # Module 2 - Top Right  
        (1, 3.5),   # Module 3 - Middle Left
        (8, 3.5),   # Module 4 - Middle Right
        (1, 0.5),   # Module 5 - Bottom Left
        (8, 0.5)    # Module 6 - Bottom Right
    ]
    
    # Module configurations
    modules = [
        {
            'title': '1. Data-Preprocessing\nModule',
            'items': [
                '• Lowercasing',
                '• Removal of URLs,',
                '  mentions, hashtags,',
                '  punctuation',
                '• Tokenization',
                '• Lemmatization'
            ]
        },
        {
            'title': '2. Emotion Classification\nModule',
            'items': [
                '• Logistic Regression / SVM',
                '• Load pre-trained model',
                '• Predict primary emotion',
                '• Return top-3 emotion',
                '  probabilities'
            ]
        },
        {
            'title': '3. Model Training &\nOptimization Module',
            'items': [
                '• Dataset loading and',
                '  preprocessing',
                '• Model compilation and',
                '  training',
                '• Hyperparameter tuning',
                '• Evaluation metrics',
                '  (accuracy, F1-score)'
            ]
        },
        {
            'title': '4. Real-time Analysis\nModule',
            'items': [
                '• Capture text input',
                '  from user',
                '• Trigger prediction using',
                '  model',
                '• Return results instantly'
            ]
        },
        {
            'title': '5. Visualization & Reporting\nModule',
            'items': [
                'Tools: Plotly, Streamlit charts',
                '• Bar and Pie chart genera-',
                '  tion',
                '• Display of confidence sco-',
                '  res'
            ]
        },
        {
            'title': '6. Evaluation & Feedback\nModule',
            'items': [
                '• Collect misclassified or',
                '  uncertain inputs',
                '• Allow user corrections/',
                '  feedback',
                '• Retrain with improved',
                '  data batches'
            ]
        }
    ]
    
    # Draw modules
    for i, (pos, module) in enumerate(zip(positions, modules)):
        x, y = pos
        
        # Create rounded rectangle box
        box = FancyBboxPatch(
            (x, y), 6, 2.5,
            boxstyle="round,pad=0.15",
            facecolor='white',
            edgecolor='#3498DB',
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add title
        ax.text(x + 3, y + 2.15, module['title'], 
                fontsize=12, fontweight='bold', ha='center', 
                color='#2C3E50', va='center')
        
        # Add bullet points
        start_y = y + 1.6
        for j, item in enumerate(module['items']):
            ax.text(x + 0.3, start_y - (j * 0.25), item, 
                    fontsize=10, ha='left', va='top', color='#34495E')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 Generating Professional Module Design Diagram...")
    fig = create_professional_module_design()
    
    # Save as high-quality PNG
    output_path = "Professional_Module_Design.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Professional Module Design saved as: {output_path}")
    print("📊 Features:")
    print("   • Clean 2x3 grid layout")
    print("   • Professional blue borders")
    print("   • Clear module descriptions")
    print("   • High-resolution (300 DPI)")
    print("   • Slide-ready format")
