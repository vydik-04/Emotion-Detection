"""
Streamlit web application for emotion detection from text
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append('..')  # Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultimate_emotion_detector import UltimateEmotionDetector
import os


# Page configuration
st.set_page_config(
    page_title="Emotion Detection from Text",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_emotion_detector():
    """Load the Ultimate Hybrid Emotion Detection model (cached for performance)"""
    try:
        # Initialize the Ultimate Emotion Detector
        detector = UltimateEmotionDetector()
        
        if detector.model is not None:
            # Create wrapper to match Streamlit app interface
            class StreamlitWrapper:
                def __init__(self, ultimate_detector):
                    self.detector = ultimate_detector
                    self.EMOTIONS = ultimate_detector.emotions
                
                def predict(self, text):
                    return self.detector.predict_single_emotion(text)
                
                def predict_proba(self, text):
                    analysis = self.detector.analyze_text(text)
                    # Convert to dictionary format expected by Streamlit app
                    probs = {}
                    for item in analysis['top_3_emotions']:
                        probs[item['emotion']] = item['probability']
                    
                    # Fill in missing emotions with very small probabilities
                    for emotion in self.EMOTIONS:
                        if emotion not in probs:
                            probs[emotion] = 0.01
                    
                    return probs
                
                def preprocess_text(self, text):
                    # Simple preprocessing for display
                    import re
                    text = text.lower()
                    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                    text = re.sub(r'@\w+|#\w+', '', text)
                    text = re.sub(r'[^a-zA-Z\s]', '', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text
            
            wrapper = StreamlitWrapper(detector)
            return wrapper, True
        else:
            return None, False
            
    except Exception as e:
        print(f"Error loading Ultimate Detector: {e}")
        return None, False


def create_emotion_chart(probabilities):
    """Create a bar chart for emotion probabilities"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Create color map for emotions
    color_map = {
        'joy': '#FFD700',
        'sadness': '#4169E1',
        'anger': '#DC143C',
        'fear': '#8B0000',
        'surprise': '#FF8C00',
        'disgust': '#9ACD32',
        'neutral': '#708090'
    }
    
    colors = [color_map.get(emotion, '#808080') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probs,
            marker_color=colors,
            text=[f'{p:.2%}' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Probabilities",
        xaxis_title="Emotions",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig


def create_emotion_pie_chart(probabilities):
    """Create a pie chart for emotion probabilities"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Filter out very small probabilities for cleaner visualization
    filtered_data = [(e, p) for e, p in zip(emotions, probs) if p > 0.05]
    if not filtered_data:
        filtered_data = [(emotions[0], probs[0])]  # Show at least one
    
    filtered_emotions, filtered_probs = zip(*filtered_data)
    
    fig = px.pie(
        values=filtered_probs,
        names=filtered_emotions,
        title="Emotion Distribution"
    )
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🎭 Emotion Detection from Text")
    st.markdown("### Analyze the emotional content of your text using AI")
    
    # Load model
    detector, model_loaded = load_emotion_detector()
    
    # Sidebar
    st.sidebar.title("About")
    
    if model_loaded and detector:
        emotion_count = len(detector.EMOTIONS)
        st.sidebar.success(f"✅ Model loaded with {emotion_count} emotions")
        
        with st.sidebar.expander("View All Emotions"):
            for i, emotion in enumerate(detector.EMOTIONS, 1):
                st.sidebar.write(f"{i}. {emotion.title()}")
    
    st.sidebar.info(
        """
        This app uses a Hybrid ML and Rule-Based system to detect 27 emotions.
        
        **Features:**
        - Real-time hybrid analysis
        - Visual charts and graphs
        - Confidence scores
        - Interactive examples
        - Negation Handling (e.g., 'not happy')
        """
    )

    # ADDED: Clear cache button
    if st.sidebar.button('🧹 Clear Cache & Reload Model'):
        st.cache_resource.clear()
        st.rerun()
    
    if not model_loaded:
        st.sidebar.warning("⚠️ Model needs to be trained first. Run `python src/train.py` to train the model.")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Text")
        
        # Single text input area with unique key for JavaScript targeting
        user_text = st.text_area(
            "Enter text to analyze (Press Ctrl+Enter to analyze):",
            placeholder="Type or paste your text here... Press Ctrl+Enter to analyze!",
            height=150,
            key="emotion_text_input"
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Emotion", type="primary", key="analyze_button")
        
        # Add JavaScript for Ctrl+Enter functionality
        st.markdown("""
        <script>
        const doc = window.parent.document;
        
        // Function to trigger analysis
        function triggerAnalysis() {
            const analyzeButton = doc.querySelector('button[data-testid="baseButton-secondary"][kind="primary"]');
            if (analyzeButton && analyzeButton.textContent.includes('Analyze Emotion')) {
                analyzeButton.click();
            }
        }
        
        // Add event listener for Ctrl+Enter in text areas
        function addKeyListener() {
            const textAreas = doc.querySelectorAll('textarea');
            textAreas.forEach(function(textarea) {
                textarea.removeEventListener('keydown', handleKeyDown); // Remove existing listeners
                textarea.addEventListener('keydown', handleKeyDown);
            });
        }
        
        function handleKeyDown(event) {
            if (event.ctrlKey && event.key === 'Enter') {
                event.preventDefault();
                triggerAnalysis();
            }
        }
        
        // Set up the listener when page loads
        if (doc.readyState === 'loading') {
            doc.addEventListener('DOMContentLoaded', addKeyListener);
        } else {
            addKeyListener();
        }
        
        // Also set up with a small delay to ensure Streamlit elements are ready
        setTimeout(addKeyListener, 1000);
        
        // Re-add listeners when Streamlit re-renders (use MutationObserver)
        const observer = new MutationObserver(function(mutations) {
            let shouldAddListener = false;
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    shouldAddListener = true;
                }
            });
            if (shouldAddListener) {
                setTimeout(addKeyListener, 100);
            }
        });
        
        observer.observe(doc.body, {
            childList: true,
            subtree: true
        });
        </script>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Quick Examples")
        example_texts = [
            "I'm so excited about my vacation!",
            "This movie made me cry so much.",
            "I can't believe they canceled the event!",
            "The dark forest at night scares me.",
            "Wow, I won the lottery!",
            "This food tastes terrible.",
            "The meeting is scheduled for 2 PM."
        ]
        
        selected_example = st.selectbox("Try an example:", [""] + example_texts)
        
        if selected_example and st.button("Use This Example"):
            user_text = selected_example
            analyze_button = True
    
    # Analysis results
    if analyze_button and user_text.strip():
        if not model_loaded:
            st.error("❌ Cannot analyze text. Model needs to be trained first.")
            st.info("Please run `python src/train.py` in the project directory to train the model.")
        else:
            with st.spinner("Analyzing emotion..."):
                try:
                    # Get predictions
                    predicted_emotion = detector.predict(user_text)
                    probabilities = detector.predict_proba(user_text)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Main result
                    st.subheader("📊 Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Emotion", predicted_emotion.title())
                    
                    with col2:
                        confidence = max(probabilities.values())
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col3:
                        processed_text = detector.preprocess_text(user_text)
                        word_count = len(processed_text.split())
                        st.metric("Processed Words", word_count)
                    
                    # Visualization
                    st.subheader("📈 Detailed Analysis")
                    
                    tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
                    
                    with tab1:
                        fig_bar = create_emotion_chart(probabilities)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with tab2:
                        fig_pie = create_emotion_pie_chart(probabilities)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Probability table
                    st.subheader("🔢 Probability Breakdown")
                    prob_df = pd.DataFrame([
                        {"Emotion": emotion.title(), "Probability": f"{prob:.1%}", "Score": prob}
                        for emotion, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(prob_df[["Emotion", "Probability"]], use_container_width=True, hide_index=True)
                    
                    # Text preprocessing info
                    with st.expander("🔧 Text Preprocessing Details"):
                        st.write("**Original Text:**")
                        st.text(user_text)
                        st.write("**Preprocessed Text:**")
                        st.text(processed_text)
                        st.write("**Preprocessing Steps:**")
                        st.markdown("""
                        1. Convert to lowercase
                        2. Remove URLs, mentions, hashtags
                        3. Remove special characters and digits
                        4. Remove extra whitespace
                        5. Remove stopwords
                        6. Lemmatize words
                        """)
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
    
    elif analyze_button and not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Built with ❤️ using Streamlit | Emotion Detection from Text
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()