"""
Systematic Data Flow Diagram Generator
Creates a clean, organized, and aesthetically pleasing data flow diagram.
"""

import matplotlib.pyplot as plt

def create_systematic_data_flow_diagram():
    """Create a systematic and clean data flow diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC') # Light background

    # Define colors and styles
    colors = {
        'process': '#D6EAF8',  # Light Blue
        'io': '#D5F5E3',       # Light Green
        'border': '#34495E',
        'arrow': '#2C3E50'
    }
    box_style = dict(boxstyle="round,pad=0.5", ec=colors['border'], lw=1.5)
    arrow_props = dict(arrowstyle="->", color=colors['arrow'], lw=2)

    # Box positions - increased vertical spacing for clarity
    positions = {
        'title': (5, 13),
        'input': (5, 11.5),
        'preprocess': (5, 10),
        'detection': (5, 8.5),
        'prediction': (5, 7),
        'visuals': (5, 5.5),
        'output': (5, 4)
    }

    # Box content
    box_data = {
        'title': {'label': 'Hybrid Emotion Analysis Data Flow', 'color': colors['process']},
        'input': {'label': 'User Input', 'color': colors['io']},
        'preprocess': {'label': 'Data Preprocessing', 'color': colors['process']},
        'detection': {'label': 'Emotion Detection Engine\n(ML + Rules)', 'color': colors['process']},
        'prediction': {'label': 'Prediction & Probability Scores', 'color': colors['process']},
        'visuals': {'label': 'Visualization Generation\n(Bar Chart & Pie Chart)', 'color': colors['process']},
        'output': {'label': 'Output to User', 'color': colors['io']},
    }

    # Draw boxes
    for key, pos in positions.items():
        data = box_data[key]
        ax.text(pos[0], pos[1], data['label'], 
                ha='center', va='center', size=11, color=colors['border'],
                bbox={**box_style, 'facecolor': data['color']})

    # Draw arrows - perfectly straight and aligned
    for i in range(len(list(positions.keys())) - 1):
        key1 = list(positions.keys())[i]
        key2 = list(positions.keys())[i+1]
        
        pos1 = positions[key1]
        pos2 = positions[key2]

        # Get the bottom of the top box and top of the bottom box
        y_start = pos1[1] - 0.5
        y_end = pos2[1] + 0.5

        # For the last box, adjust arrow position slightly
        if key2 == 'output':
             y_end = pos2[1] + 0.6

        ax.annotate("", xy=(pos1[0], y_end), xytext=(pos1[0], y_start), arrowprops=arrow_props)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 Generating Systematic Data Flow Diagram...")
    fig = create_systematic_data_flow_diagram()
    output_path = "Systematic_Data_Flow_Diagram.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
    print(f"✅ Diagram saved as: {output_path}")
