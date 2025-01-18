# app.py
import streamlit as st
import torch
from pathlib import Path
from models.model import HateSpeechClassifier
from models.classification_heads import MLPHead
from utils.predictor import HateSpeechPredictor
from training.trainer import Trainer
import plotly.graph_objects as go
import pandas as pd

class App:
    def __init__(self):
        self.config = {
            "run_name": "mlp_head",
            "model_name": "microsoft/deberta-v3-base",
            "num_classes": 2,
            "batch_size": 32,
            "max_length": 128,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        self.model = self.load_model()
        self.predictor = HateSpeechPredictor(self.model, self.config["model_name"])

    def load_model(self):
        # Initialize model architecture
        classification_head = MLPHead(768, 1536, 384, self.config["num_classes"])
        model = HateSpeechClassifier(
            self.config["model_name"],
            classification_head,
            freeze_bert=False
        ).to(self.config["device"])
        
        # Load best checkpoint
        best_checkpoint = list(Path('checkpoints').glob('best_model_*.pt'))[0]
        trainer = Trainer(
            model=model,
            optimizer=None,  # Not needed for inference
            criterion=None,  # Not needed for inference
            device=self.config["device"]
        )
        trainer.load_checkpoint(str(best_checkpoint))
        
        return model

    def create_prediction_chart(self, probabilities):
        fig = go.Figure(data=[
            go.Bar(
                x=['Non-toxic', 'Toxic'],
                y=[probabilities['non_toxic_probability'], probabilities['toxic_probability']],
                marker_color=['green', 'red']
            )
        ])
        
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            showlegend=False
        )
        
        return fig

def main():
    st.set_page_config(
        page_title="Hate Speech Detection",
        page_icon="🔍",
        layout="wide"
    )

    # Initialize app
    app = App()

    # Title and description
    st.title("🔍 Hate Speech Detection")
    st.markdown("""
    This application uses a DeBERTa-based model to detect toxic content in text.
    Enter your text below to analyze it for potential hate speech or toxic content.
    """)

    # Create tabs
    tab1, tab2 = st.tabs(["Single Text Analysis", "Batch Analysis"])

    # Single Text Analysis Tab
    with tab1:
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste text here..."
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("Analyze Text", key="single_analyze"):
                if text_input.strip():
                    with st.spinner('Analyzing...'):
                        result = app.predictor.predict([text_input])[0]
                        
                        # Display prediction
                        prediction = "Toxic" if result['toxic_probability'] > 0.5 else "Non-toxic"
                        prediction_color = "red" if prediction == "Toxic" else "green"
                        
                        st.markdown(f"### Prediction: <span style='color:{prediction_color}'>{prediction}</span>", 
                                  unsafe_allow_html=True)
                        
                        # Display confidence
                        confidence = max(result['toxic_probability'], result['non_toxic_probability'])
                        st.markdown(f"### Confidence: {confidence:.2%}")

        with col2:
            if text_input.strip() and 'result' in locals():
                # Create and display probability chart
                fig = app.create_prediction_chart(result)
                st.plotly_chart(fig, use_container_width=True)

    # Batch Analysis Tab
    with tab2:
        st.markdown("### Batch Analysis")
        
        # Text area for batch input
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="Enter one text per line..."
        )

        if st.button("Analyze Batch", key="batch_analyze"):
            if batch_input.strip():
                texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner('Analyzing batch...'):
                        results = app.predictor.predict(texts)
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            'Text': texts,
                            'Prediction': ['Toxic' if r['toxic_probability'] > 0.5 else 'Non-toxic' for r in results],
                            'Toxic Probability': [r['toxic_probability'] for r in results],
                            'Non-toxic Probability': [r['non_toxic_probability'] for r in results]
                        })
                        
                        # Display results
                        st.markdown("### Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Display statistics
                        st.markdown("### Batch Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            toxic_count = sum(1 for r in results if r['toxic_probability'] > 0.5)
                            st.metric("Toxic Texts", toxic_count)
                            
                        with col2:
                            non_toxic_count = len(texts) - toxic_count
                            st.metric("Non-toxic Texts", non_toxic_count)
                else:
                    st.warning("Please enter at least one text to analyze.")

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Model: DeBERTa-v3-base fine-tuned on ToxiGen dataset</p>
        <p>Note: This is an experimental tool. Results should be interpreted with caution.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()