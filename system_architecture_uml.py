"""
System Architecture UML Diagram Generator
Creates a comprehensive UML diagram for the Emotion Detection system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_architecture_uml():
    """Create a comprehensive UML diagram for the emotion detection system"""
    
    # Set up the figure with high DPI for better quality
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors for different components
    colors = {
        'ui': '#E3F2FD',           # Light Blue
        'core': '#F3E5F5',         # Light Purple  
        'ml': '#E8F5E8',           # Light Green
        'data': '#FFF3E0',         # Light Orange
        'memory': '#FCE4EC',       # Light Pink
        'border': '#424242'        # Dark Gray
    }
    
    # Title
    ax.text(8, 11.5, 'Emotion Detection System - UML Architecture Diagram', 
            fontsize=20, fontweight='bold', ha='center')
    
    # 1. User Interface Layer (Top)
    ui_box = FancyBboxPatch((0.5, 9.5), 7, 1.5, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['ui'], 
                           edgecolor=colors['border'], 
                           linewidth=2)
    ax.add_patch(ui_box)
    ax.text(4, 10.7, '<<User Interface Layer>>', fontsize=12, fontweight='bold', ha='center')
    ax.text(4, 10.3, 'Streamlit Web Application', fontsize=11, ha='center')
    ax.text(4, 10.0, '• Text Input Interface', fontsize=9, ha='center')
    ax.text(4, 9.7, '• Interactive Charts & Visualizations', fontsize=9, ha='center')
    
    # 2. Core Processing Layer (Middle)
    core_box = FancyBboxPatch((0.5, 6.5), 7, 2.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['core'], 
                             edgecolor=colors['border'], 
                             linewidth=2)
    ax.add_patch(core_box)
    ax.text(4, 8.7, '<<Core Processing Layer>>', fontsize=12, fontweight='bold', ha='center')
    ax.text(4, 8.3, 'Ultimate Emotion Detector', fontsize=11, fontweight='bold', ha='center')
    
    # Core components
    ax.text(2, 7.9, '• Text Preprocessing', fontsize=9, ha='left')
    ax.text(2, 7.6, '• ML Model Inference', fontsize=9, ha='left')
    ax.text(2, 7.3, '• Rule-Based Enhancement', fontsize=9, ha='left')
    ax.text(2, 7.0, '• Emotion Analysis Engine', fontsize=9, ha='left')
    ax.text(2, 6.7, '• Top-3 Emotion Selection', fontsize=9, ha='left')
    
    # 3. Machine Learning Models (Right)
    ml_box = FancyBboxPatch((8.5, 6.5), 7, 2.5, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['ml'], 
                           edgecolor=colors['border'], 
                           linewidth=2)
    ax.add_patch(ml_box)
    ax.text(12, 8.7, '<<ML Models Repository>>', fontsize=12, fontweight='bold', ha='center')
    
    # ML Model details
    ax.text(9, 8.3, '• Logistic Regression (99.23% accuracy)', fontsize=9, ha='left')
    ax.text(9, 8.0, '• SVM (Linear & RBF kernels)', fontsize=9, ha='left')
    ax.text(9, 7.7, '• Random Forest Ensemble', fontsize=9, ha='left')
    ax.text(9, 7.4, '• TF-IDF Vectorizer (100K features)', fontsize=9, ha='left')
    ax.text(9, 7.1, '• Label Encoder (27 emotions)', fontsize=9, ha='left')
    ax.text(9, 6.8, '• Ultra High Accuracy Model', fontsize=9, ha='left')
    
    # 4. Data Layer (Bottom Left)
    data_box = FancyBboxPatch((0.5, 3.5), 7, 2.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['data'], 
                             edgecolor=colors['border'], 
                             linewidth=2)
    ax.add_patch(data_box)
    ax.text(4, 5.7, '<<Data Layer>>', fontsize=12, fontweight='bold', ha='center')
    
    # Data components
    ax.text(1, 5.3, '• GoEmotions Dataset (1.35M samples)', fontsize=9, ha='left')
    ax.text(1, 5.0, '• Hugging Face Emotion Dataset', fontsize=9, ha='left')
    ax.text(1, 4.7, '• Tweet Emotions Dataset', fontsize=9, ha='left')
    ax.text(1, 4.4, '• Custom Augmented Data', fontsize=9, ha='left')
    ax.text(1, 4.1, '• Training & Evaluation Scripts', fontsize=9, ha='left')
    ax.text(1, 3.8, '• Model Performance Metrics', fontsize=9, ha='left')
    
    # 5. Memory Optimized System (Bottom Right)
    memory_box = FancyBboxPatch((8.5, 3.5), 7, 2.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['memory'], 
                               edgecolor=colors['border'], 
                               linewidth=2)
    ax.add_patch(memory_box)
    ax.text(12, 5.7, '<<Memory Optimized System>>', fontsize=12, fontweight='bold', ha='center')
    
    # Memory optimization features
    ax.text(9, 5.3, '• Aggressive/Balanced/Conservative modes', fontsize=9, ha='left')
    ax.text(9, 5.0, '• Batch Processing (1K-5K samples)', fontsize=9, ha='left')
    ax.text(9, 4.7, '• Reduced Feature Sets (1K-5K)', fontsize=9, ha='left')
    ax.text(9, 4.4, '• Memory-Efficient Vectorization', fontsize=9, ha='left')
    ax.text(9, 4.1, '• Optimized Ensemble Methods', fontsize=9, ha='left')
    ax.text(9, 3.8, '• Garbage Collection Integration', fontsize=9, ha='left')
    
    # 6. Rule-Based Enhancement (Center)
    rule_box = FancyBboxPatch((3, 1), 10, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFFDE7', 
                             edgecolor=colors['border'], 
                             linewidth=2)
    ax.add_patch(rule_box)
    ax.text(8, 2.2, '<<Rule-Based Enhancement Engine>>', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 1.8, '27 Emotion-Specific Keyword Rules • Pattern Matching • Context Analysis', fontsize=10, ha='center')
    ax.text(8, 1.4, 'Priority Handling • Sexual_desire • Boredom • Horror • Nostalgia • Confusion', fontsize=9, ha='center')
    
    # Add arrows showing data flow
    arrows = [
        # UI to Core
        ((4, 9.5), (4, 9.0)),
        # Core to ML Models
        ((7.5, 7.7), (8.5, 7.7)),
        # Core to Data
        ((4, 6.5), (4, 6.0)),
        # Memory System to Core (optional)
        ((8.5, 4.7), (7.5, 6.8)),
        # Rule Engine to Core
        ((8, 2.5), (4, 6.5))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=colors['border'], 
                              ec=colors['border'], linewidth=2)
        ax.add_patch(arrow)
    
    # Add emotion categories legend
    legend_box = FancyBboxPatch((13, 0.2), 2.7, 2.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F5F5F5', 
                               edgecolor=colors['border'], 
                               linewidth=1)
    ax.add_patch(legend_box)
    ax.text(14.35, 2.5, 'Emotion Categories', fontsize=10, fontweight='bold', ha='center')
    ax.text(13.1, 2.2, 'Basic 7:', fontsize=8, fontweight='bold', ha='left')
    ax.text(13.1, 2.0, 'joy, sadness, anger', fontsize=7, ha='left')
    ax.text(13.1, 1.8, 'fear, surprise, disgust, neutral', fontsize=7, ha='left')
    ax.text(13.1, 1.5, 'Extended 27:', fontsize=8, fontweight='bold', ha='left')
    ax.text(13.1, 1.3, 'admiration, awe, romance', fontsize=7, ha='left')
    ax.text(13.1, 1.1, 'empathic_pain, confusion', fontsize=7, ha='left')
    ax.text(13.1, 0.9, 'sexual_desire, nostalgia', fontsize=7, ha='left')
    ax.text(13.1, 0.7, 'boredom, horror, etc.', fontsize=7, ha='left')
    
    # Add performance metrics
    perf_box = FancyBboxPatch((0.2, 0.2), 4, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#E8F5E8', 
                             edgecolor=colors['border'], 
                             linewidth=1)
    ax.add_patch(perf_box)
    ax.text(2.2, 1.2, 'Performance Metrics', fontsize=10, fontweight='bold', ha='center')
    ax.text(0.4, 0.9, '• Ultra High Accuracy Model: 99.23%', fontsize=8, ha='left')
    ax.text(0.4, 0.7, '• Large Scale Model: 50.13% (1.35M samples)', fontsize=8, ha='left')
    ax.text(0.4, 0.5, '• Memory Optimized: Variable (Mode dependent)', fontsize=8, ha='left')
    ax.text(0.4, 0.3, '• Real-time Processing: <1 second', fontsize=8, ha='left')
    
    # Add deployment info
    deploy_box = FancyBboxPatch((5, 0.2), 3.5, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFF3E0', 
                               edgecolor=colors['border'], 
                               linewidth=1)
    ax.add_patch(deploy_box)
    ax.text(6.75, 1.2, 'Deployment Options', fontsize=10, fontweight='bold', ha='center')
    ax.text(5.2, 0.9, '• Streamlit Web App (app.py)', fontsize=8, ha='left')
    ax.text(5.2, 0.7, '• Flask API Support', fontsize=8, ha='left')
    ax.text(5.2, 0.5, '• Memory-Optimized for Edge', fontsize=8, ha='left')
    ax.text(5.2, 0.3, '• Production-Ready Models', fontsize=8, ha='left')
    
    plt.tight_layout()
    return fig

# Generate and save the diagram
if __name__ == "__main__":
    print("🎨 Generating UML System Architecture Diagram...")
    fig = create_system_architecture_uml()
    
    # Save as high-quality PNG
    output_path = "System_Architecture_UML_Diagram.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ UML Diagram saved as: {output_path}")
    print("📊 Diagram Features:")
    print("   • High-resolution (300 DPI)")
    print("   • Complete system overview")
    print("   • Component relationships")
    print("   • Performance metrics")
    print("   • Deployment information")
    
    # Display the plot
    plt.show()
