"""
Data Flow Diagram Generator
Creates a clean, colorful data flow diagram inspired by the user's reference image.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_data_flow_diagram():
    """Create a data flow diagram with a specific blue and green theme."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12)
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

    # Box positions
    positions = {
        'title': (4, 11),
        'input': (4, 9.5),
        'preprocess': (4, 8),
        'detection': (4, 6.5),
        'prediction': (4, 5),
        'bar_chart': (2.5, 3),
        'pie_chart': (5.5, 3),
        'output': (4, 1)
    }

    # Create boxes
    ax.text(positions['title'][0], positions['title'][1], 
            'Hybrid Emotion Analysis Data Flow', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['process']})
    
    ax.text(positions['input'][0], positions['input'][1], 
            'User Input', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['io']})

    ax.text(positions['preprocess'][0], positions['preprocess'][1], 
            'Preprocessing', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['process']})

    ax.text(positions['detection'][0], positions['detection'][1], 
            'Hybrid Emotion Detection\n(ML + Rules)', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['process']})

    ax.text(positions['prediction'][0], positions['prediction'][1], 
            'Prediction & Probability Scores', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['process']})
    
    ax.text(positions['bar_chart'][0], positions['bar_chart'][1], 
            'Bar Chart', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['process']})
    
    ax.text(positions['pie_chart'][0], positions['pie_chart'][1], 
            'Pie Chart', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['process']})
    
    ax.text(positions['output'][0], positions['output'][1], 
            'Output to User', 
            ha='center', va='center', size=11, color=colors['border'],
            bbox={**box_style, 'facecolor': colors['io']})

    # Connect boxes with arrows
    ax.annotate("", xy=(4, 10.5), xytext=(4, 11.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 9), xytext=(4, 10), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 7.5), xytext=(4, 8.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 6), xytext=(4, 7), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 4.5), xytext=(4, 5.5), arrowprops=arrow_props)
    
    # Forking arrows
    ax.annotate("", xy=(2.5, 3.5), xytext=(4, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(5.5, 3.5), xytext=(4, 4.5), arrowprops=arrow_props)
    
    # Merging arrows
    ax.annotate("", xy=(4, 1.5), xytext=(2.5, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 1.5), xytext=(5.5, 2.5), arrowprops=arrow_props)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 Generating Data Flow Diagram...")
    fig = create_data_flow_diagram()
    output_path = "Data_Flow_Diagram.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Diagram saved as: {output_path}")
