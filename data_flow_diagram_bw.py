import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

def create_data_flow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Data Flow Diagram - Emotion Detection System', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Define styles for different components
    external_style = {'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 2}
    process_style = {'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.5}
    datastore_style = {'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.5}
    
    # External Entities (Squares)
    user_box = Rectangle((0.5, 7.5), 2, 1, **external_style)
    ax.add_patch(user_box)
    ax.text(1.5, 8, 'User', ha='center', va='center', fontsize=10, fontweight='bold')
    
    app_box = Rectangle((9.5, 7.5), 2, 1, **external_style)
    ax.add_patch(app_box)
    ax.text(10.5, 8, 'Application\nSystem', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Processes (Circles)
    # Process 1: Text Input Processing
    process1 = Circle((3.5, 8), 0.8, **process_style)
    ax.add_patch(process1)
    ax.text(3.5, 8, '1\nText\nProcessing', ha='center', va='center', fontsize=9)
    
    # Process 2: ML Classification
    process2 = Circle((3.5, 5.5), 0.8, **process_style)
    ax.add_patch(process2)
    ax.text(3.5, 5.5, '2\nML\nClassification', ha='center', va='center', fontsize=9)
    
    # Process 3: Rule-Based Analysis
    process3 = Circle((6.5, 5.5), 0.8, **process_style)
    ax.add_patch(process3)
    ax.text(6.5, 5.5, '3\nRule-Based\nAnalysis', ha='center', va='center', fontsize=9)
    
    # Process 4: Hybrid Decision
    process4 = Circle((5, 3), 0.8, **process_style)
    ax.add_patch(process4)
    ax.text(5, 3, '4\nHybrid\nDecision', ha='center', va='center', fontsize=9)
    
    # Process 5: Result Generation
    process5 = Circle((8.5, 3), 0.8, **process_style)
    ax.add_patch(process5)
    ax.text(8.5, 3, '5\nResult\nGeneration', ha='center', va='center', fontsize=9)
    
    # Data Stores (Open rectangles)
    # D1: Training Dataset
    d1_line1 = Rectangle((1, 2), 2.5, 0.05, **datastore_style)
    d1_line2 = Rectangle((1, 1.4), 2.5, 0.05, **datastore_style)
    ax.add_patch(d1_line1)
    ax.add_patch(d1_line2)
    ax.text(0.8, 1.7, 'D1', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.25, 1.7, 'Training Dataset\n(Emotion Labels)', ha='center', va='center', fontsize=9)
    
    # D2: ML Models
    d2_line1 = Rectangle((6.5, 2), 2.5, 0.05, **datastore_style)
    d2_line2 = Rectangle((6.5, 1.4), 2.5, 0.05, **datastore_style)
    ax.add_patch(d2_line1)
    ax.add_patch(d2_line2)
    ax.text(6.3, 1.7, 'D2', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.75, 1.7, 'Trained Models\n(LR, SVM)', ha='center', va='center', fontsize=9)
    
    # D3: Rule Knowledge Base
    d3_line1 = Rectangle((9.5, 5), 2.5, 0.05, **datastore_style)
    d3_line2 = Rectangle((9.5, 4.4), 2.5, 0.05, **datastore_style)
    ax.add_patch(d3_line1)
    ax.add_patch(d3_line2)
    ax.text(9.3, 4.7, 'D3', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(10.75, 4.7, 'Emotion Rules\n(27 Patterns)', ha='center', va='center', fontsize=9)
    
    # Data Flows (Arrows with labels)
    arrow_style = dict(arrowstyle='->', lw=1.5, color='black')
    
    # User to Process 1
    ax.annotate('', xy=(2.7, 8), xytext=(2.5, 8),
                arrowprops=arrow_style)
    ax.text(2.1, 8.3, 'Raw Text', fontsize=8, ha='center')
    
    # Process 1 to Process 2
    ax.annotate('', xy=(3.5, 6.3), xytext=(3.5, 7.2),
                arrowprops=arrow_style)
    ax.text(2.8, 6.8, 'Preprocessed\nText', fontsize=8, ha='center')
    
    # Process 1 to Process 3
    ax.annotate('', xy=(5.8, 6.1), xytext=(4.2, 7.4),
                arrowprops=arrow_style)
    ax.text(5.5, 7, 'Original\nText', fontsize=8, ha='center')
    
    # Process 2 to Process 4
    ax.annotate('', xy=(4.3, 3.6), xytext=(4, 4.8),
                arrowprops=arrow_style)
    ax.text(3.5, 4.2, 'ML\nPredictions', fontsize=8, ha='center')
    
    # Process 3 to Process 4
    ax.annotate('', xy=(5.7, 3.6), xytext=(6, 4.8),
                arrowprops=arrow_style)
    ax.text(6.5, 4.2, 'Rule\nMatches', fontsize=8, ha='center')
    
    # Process 4 to Process 5
    ax.annotate('', xy=(7.7, 3), xytext=(5.8, 3),
                arrowprops=arrow_style)
    ax.text(6.75, 3.3, 'Final\nEmotion', fontsize=8, ha='center')
    
    # Process 5 to Application
    ax.annotate('', xy=(9.5, 7.5), xytext=(8.8, 3.7),
                arrowprops=arrow_style)
    ax.text(9.8, 5.5, 'Classification\nResults', fontsize=8, ha='center', rotation=70)
    
    # Data Store connections
    # D1 to Process 2 (Training data)
    ax.annotate('', xy=(3, 4.8), xytext=(2.5, 2.1),
                arrowprops=arrow_style)
    ax.text(2.2, 3.5, 'Training\nData', fontsize=8, ha='center', rotation=60)
    
    # Process 2 to D2 (Model storage)
    ax.annotate('', xy=(7, 2.1), xytext=(4, 4.8),
                arrowprops=arrow_style)
    ax.text(5.5, 3.2, 'Trained\nModel', fontsize=8, ha='center', rotation=-30)
    
    # D2 to Process 2 (Model loading)
    ax.annotate('', xy=(4, 4.8), xytext=(7, 2.1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black', linestyle='dashed'))
    
    # D3 to Process 3 (Rule access)
    ax.annotate('', xy=(7.2, 5.1), xytext=(9.5, 4.7),
                arrowprops=arrow_style)
    ax.text(8.5, 5.2, 'Rule\nPatterns', fontsize=8, ha='center')
    
    # Add legend
    legend_y = 0.5
    ax.text(0.5, legend_y + 0.6, 'Legend:', fontsize=10, fontweight='bold')
    
    # External entity
    legend_ext = Rectangle((0.5, legend_y + 0.3), 0.4, 0.2, **external_style)
    ax.add_patch(legend_ext)
    ax.text(1.1, legend_y + 0.4, 'External Entity', fontsize=8)
    
    # Process
    legend_proc = Circle((0.7, legend_y), 0.1, **process_style)
    ax.add_patch(legend_proc)
    ax.text(1.1, legend_y, 'Process', fontsize=8)
    
    # Data store
    legend_ds1 = Rectangle((0.5, legend_y - 0.35), 0.4, 0.02, **datastore_style)
    legend_ds2 = Rectangle((0.5, legend_y - 0.45), 0.4, 0.02, **datastore_style)
    ax.add_patch(legend_ds1)
    ax.add_patch(legend_ds2)
    ax.text(1.1, legend_y - 0.4, 'Data Store', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('emotion_data_flow_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Data Flow Diagram created successfully!")
    print("📁 Saved as: emotion_data_flow_diagram.png")

if __name__ == "__main__":
    create_data_flow_diagram()
