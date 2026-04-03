import graphviz

# This script generates a diagram of the model architecture.
# To run this, you need to have graphviz installed: pip install graphviz
# You also need to install the Graphviz software on your system: https://graphviz.org/download/

dot = graphviz.Digraph(comment='Emotion Detection Model Architecture')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2', label='Emotion Detection Model Architecture', labelloc='t', fontsize='16')

# Define node styles
node_style = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#EFEFEF',
    'fontname': 'Helvetica',
    'fontsize': '11'
}
process_style = {
    'shape': 'box',
    'style': 'filled',
    'fillcolor': '#D6EAF8',
    'fontname': 'Helvetica',
    'fontsize': '11'
}
model_style = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': '#D5F5E3',
    'fontname': 'Helvetica',
    'fontsize': '11'
}
logic_style = {
    'shape': 'diamond',
    'style': 'filled',
    'fillcolor': '#FAD7A0',
    'fontname': 'Helvetica',
    'fontsize': '11',
    'fixedsize': 'true',
    'width': '2.5',
    'height': '1.5'
}
output_style = {
    'shape': 'ellipse',
    'style': 'filled',
    'fillcolor': '#FDEDEC',
    'fontname': 'Helvetica',
    'fontsize': '11'
}

# --- Define Nodes ---
dot.node('input', 'Input\n(Raw Text)', **node_style)

# ML Path Subgraph
with dot.subgraph(name='cluster_ml') as c:
    c.attr(label='Machine Learning Path', style='rounded', color='grey', fontname='Helvetica')
    c.node('preprocess', '1. Negation-Aware Preprocessing\n(Tokenize, Lemmatize, Clean)', **process_style)
    c.node('vectorize', '2. TF-IDF Vectorization\n(Text to Numerical Features)', **process_style)
    c.node('ml_model', '3. Trained ML Model\n(Logistic Regression / SVM)', **model_style)
    c.node('ml_output', 'Initial ML Prediction\n& Probabilities', **node_style)

# Rule-Based Path Subgraph
with dot.subgraph(name='cluster_rules') as c:
    c.attr(label='Rule-Based Path', style='rounded', color='grey', fontname='Helvetica')
    c.node('rules', 'Keyword Matching Logic\n(Search for 27 emotion keywords & patterns)', **process_style)

# Hybrid Logic and Output
dot.node('hybrid_logic', 'Hybrid Decision Logic', **logic_style)
dot.node('output', 'Final Emotion Classification\n(Top 3 emotions with scores)', **output_style)

# --- Define Edges ---

# ML Path Edges
dot.edge('input', 'preprocess')
dot.edge('preprocess', 'vectorize')
dot.edge('vectorize', 'ml_model')
dot.edge('ml_model', 'ml_output')

# Rule-Based Path Edge
dot.edge('input', 'rules', lhead='cluster_rules')

# Connect paths to hybrid logic
dot.edge('ml_output', 'hybrid_logic', label='ML Results')
dot.edge('rules', 'hybrid_logic', label='Rule Match')

# Connect logic to output
dot.edge('hybrid_logic', 'output', label='Final Decision')

# Render the graph
try:
    file_path = 'system_architecture'
    dot.render(file_path, format='png', cleanup=True)
    print(f"✅ Diagram successfully generated and saved as '{file_path}.png'")
except Exception as e:
    print(f"❌ Error generating diagram: {e}")
    print("Please ensure you have Graphviz installed on your system (see https://graphviz.org/download/) and in your Python environment (`pip install graphviz`).")

