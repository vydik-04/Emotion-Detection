"""
Module Design Diagram Generator
Creates a professional, slide-like module design diagram for the Emotion Detection system.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_module_design_diagram():
    """Create a slide-like module design diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#F0F2F5')  # Light gray background

    # Title
    ax.text(8, 8.5, 'Emotion Detection System: Module Design', 
            fontsize=24, fontweight='bold', ha='center', color='#1C3A5A')

    # Define colors and styles
    colors = {
        'ui': '#D6EAF8', 'core': '#D1F2EB', 'data': '#FDEDEC', 'ml': '#E8DAEF', 
        'rules': '#FEF9E7', 'utils': '#FDEBD0', 'border': '#2C3E50'
    }
    bbox_style = "round,pad=0.4,rounding_size=0.2"

    # 1. User Interface Module
    ui_box = FancyBboxPatch((0.5, 6), 5, 2, boxstyle=bbox_style, 
                           facecolor=colors['ui'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(ui_box)
    ax.text(3, 7.7, 'User Interface', fontsize=14, fontweight='bold', ha='center')
    ax.text(3, 7.2, 'Handles user interaction and visualization.', fontsize=9, ha='center', wrap=True)
    ax.text(1, 6.3, '• Streamlit Web App (`app.py`)\n• Plotly Charts (Bar & Pie)\n• User Input Handling', fontsize=9, ha='left')

    # 2. Core Detection Module
    core_box = FancyBboxPatch((5.7, 3.5), 4.6, 3, boxstyle=bbox_style, 
                           facecolor=colors['core'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(core_box)
    ax.text(8, 6.2, 'Core Detection Engine', fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 5.7, 'Orchestrates the analysis pipeline.', fontsize=9, ha='center', wrap=True)
    ax.text(6.2, 3.8, '• `UltimateEmotionDetector`\n• `MemoryOptimizedModel`\n• Manages ML & Rule-based systems\n• Aggregates results', fontsize=9, ha='left')

    # 3. Data Processing Module
    data_box = FancyBboxPatch((0.5, 3.5), 5, 2, boxstyle=bbox_style, 
                           facecolor=colors['data'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(3, 5.2, 'Data Processing', fontsize=14, fontweight='bold', ha='center')
    ax.text(3, 4.7, 'Handles data loading and prep.', fontsize=9, ha='center', wrap=True)
    ax.text(1, 3.8, '• Dataset Loading (CSV, Hugging Face)\n• Text Preprocessing & Cleaning\n• Memory-Efficient Batching', fontsize=9, ha='left')

    # 4. Machine Learning Module
    ml_box = FancyBboxPatch((10.5, 3.5), 5, 3.5, boxstyle=bbox_style, 
                           facecolor=colors['ml'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(ml_box)
    ax.text(13, 6.7, 'Machine Learning Models', fontsize=14, fontweight='bold', ha='center')
    ax.text(13, 6.2, 'Performs ML-based predictions.', fontsize=9, ha='center', wrap=True)
    ax.text(11, 3.8, '• `EmotionDetector` class\n• TF-IDF Vectorization\n• Classifiers (Logistic Regression, SVM)\n• Model Training & Evaluation\n• Serialization (`.joblib`)', fontsize=9, ha='left')

    # 5. Rule-Based Module
    rule_box = FancyBboxPatch((0.5, 1), 7, 2, boxstyle=bbox_style, 
                           facecolor=colors['rules'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(rule_box)
    ax.text(4, 2.7, 'Rule-Based System', fontsize=14, fontweight='bold', ha='center')
    ax.text(4, 2.2, 'Enhances predictions with keyword logic.', fontsize=9, ha='center', wrap=True)
    ax.text(1, 1.3, '• Keyword Matching Engine\n• 27 Emotion-Specific Rule Sets\n• Context & Priority Analysis', fontsize=9, ha='left')

    # 6. Utilities Module
    utils_box = FancyBboxPatch((8, 1), 7.5, 2, boxstyle=bbox_style, 
                           facecolor=colors['utils'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(utils_box)
    ax.text(11.75, 2.7, 'Utilities & Persistence', fontsize=14, fontweight='bold', ha='center')
    ax.text(11.75, 2.2, 'Provides support functions and storage.', fontsize=9, ha='center', wrap=True)
    ax.text(8.5, 1.3, '• Model & Vectorizer Persistence (`.joblib`)\n• Performance Metrics Logging (JSON)\n• Configuration Management', fontsize=9, ha='left')

    # Arrows
    def draw_arrow(start, end):
        ax.annotate('', xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle='->', lw=2, color='#4A5568'))

    draw_arrow((3, 6), (7.5, 6.5)) # UI -> Core
    draw_arrow((8, 3.5), (5, 3.2)) # Core -> Rules
    draw_arrow((8, 3.5), (12.5, 3.2)) # Core -> Utils
    draw_arrow((10.5, 5), (10.3, 4.5)) # ML -> Core
    draw_arrow((3, 3.5), (6, 4)) # Data -> Core
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 Generating Module Design Diagram...")
    fig = create_module_design_diagram()
    output_path = "Module_Design_Diagram.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#F0F2F5')
    print(f"✅ Diagram saved as: {output_path}")
