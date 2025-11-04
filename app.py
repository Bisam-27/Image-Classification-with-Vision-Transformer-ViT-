import streamlit as st
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import io
import os

# Page config
st.set_page_config(
    page_title="ViT Food Classifier",
    page_icon="🍔",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default Food-41 class names
FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries'
]

# Header
st.markdown('<div class="main-header">🍔 ViT Food Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload food images and get AI-powered predictions using Vision Transformer</div>', unsafe_allow_html=True)

# Sidebar - Model Configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Device info
    device_name = "🎮 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.info(f"Device: **{device_name}**")
    
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    
    st.divider()
    
    # Model loading section
    st.subheader("📂 Load Model")
    
    # Default to your local model path
    default_path = r"C:\Users\BISAM AHMAD\Downloads\q3-finetunningnadeem"
    
    model_path = st.text_input(
        "Model Directory Path:",
        value=default_path,
        help="Enter the path to your fine-tuned model directory"
    )
    
    # Check if path exists
    if model_path and os.path.exists(model_path):
        st.success(f"✅ Path exists: {model_path}")
        
        # Check for required files
        required_files = ['config.json', 'pytorch_model.bin']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            st.warning(f"⚠️ Missing files: {', '.join(missing_files)}")
    elif model_path:
        st.error(f"❌ Path does not exist: {model_path}")
    
    # Custom class names
    use_custom_classes = st.checkbox("Use custom class names", value=False)
    
    if use_custom_classes:
        custom_classes_text = st.text_area(
            "Class Names (one per line):",
            value="\n".join(FOOD_CLASSES),
            height=150,
            help="Enter your class names, one per line"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load Model"):
            if model_path and os.path.exists(model_path):
                with st.spinner("Loading model..."):
                    try:
                        # Load processor and model
                        st.session_state.processor = ViTImageProcessor.from_pretrained(model_path)
                        st.session_state.model = ViTForImageClassification.from_pretrained(model_path)
                        st.session_state.model.to(st.session_state.device)
                        st.session_state.model.eval()
                        
                        # Load class names
                        if use_custom_classes:
                            st.session_state.class_names = [
                                line.strip() for line in custom_classes_text.split('\n') 
                                if line.strip()
                            ]
                        else:
                            # Try to load from saved JSON file
                            class_names_path = os.path.join(model_path, 'class_names.json')
                            if os.path.exists(class_names_path):
                                with open(class_names_path, 'r') as f:
                                    st.session_state.class_names = json.load(f)
                            # Try to get from model config
                            elif hasattr(st.session_state.model.config, 'id2label'):
                                st.session_state.class_names = [
                                    st.session_state.model.config.id2label[i] 
                                    for i in range(len(st.session_state.model.config.id2label))
                                ]
                            else:
                                st.session_state.class_names = FOOD_CLASSES
                        
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
                        st.error("Make sure you've saved the model correctly using model.save_pretrained()")
            else:
                st.warning("⚠️ Please enter a valid model path")
    
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.model = None
            st.session_state.processor = None
            st.success("Model cleared!")
            st.rerun()
    
    # Model status
    st.divider()
    if st.session_state.model is not None:
        st.success("✅ Model: **Loaded**")
        
        with st.expander("📊 Model Info"):
            num_params = sum(p.numel() for p in st.session_state.model.parameters())
            st.write(f"**Parameters:** {num_params:,}")
            st.write(f"**Classes:** {st.session_state.model.config.num_labels}")
            st.write(f"**Image Size:** 224x224")
            st.write(f"**Device:** {st.session_state.device}")
    else:
        st.warning("⚠️ Model: **Not Loaded**")
    
    st.divider()
    
    # Settings
    st.subheader("🎛️ Prediction Settings")
    
    top_k = st.slider(
        "Top-K Predictions:",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top predictions to show"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold (%):",
        min_value=0,
        max_value=100,
        value=10,
        help="Minimum confidence to display"
    )
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.predictions_history = []
        st.success("History cleared!")
        st.rerun()

# Main content
if st.session_state.model is None:
    st.warning("⚠️ Please load a model from the sidebar to begin classification.")
    
    with st.expander("📖 How to use this app"):
        st.markdown("""
        ### Getting Started
        1. **Save your model first** (if you haven't already):
           - Open your Jupyter notebook: `q3-finetunningnadeem.ipynb`
           - Add the save code at the end (see documentation)
           - Run it to save your model
        
        2. **Load your model** from the sidebar:
           - Default path is already set to: `C:\\Users\\BISAM AHMAD\\Downloads\\vit-food-model`
           - Click "Load Model" button
        
        3. **Upload food images** (JPG, PNG, JPEG)
        
        4. **View predictions** with confidence scores
        
        ### Required Model Files
        Your model directory should contain:
        - `config.json` - Model configuration
        - `pytorch_model.bin` or `model.safetensors` - Model weights
        - `preprocessor_config.json` - Image processor config
        - `class_names.json` (optional) - Class labels
        
        ### Tips
        - Upload clear, well-lit food images for best results
        - Multiple images can be processed in batch mode
        - Adjust Top-K to see more or fewer predictions
        - Use confidence threshold to filter low-confidence predictions
        """)

else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Single Image", "📁 Batch Upload", "📊 History", "📈 Analytics"])
    
    # Tab 1: Single Image Classification
    with tab1:
        st.header("Single Image Classification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            
            uploaded_file = st.file_uploader(
                "Choose a food image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of food"
            )
            
            # Sample images option
            use_sample = st.checkbox("Or use a sample image")
            
            if use_sample:
                st.info("📝 Note: This is a sample colored image for testing")
                # Create a sample colored image
                sample_img = Image.new('RGB', (224, 224), color=(255, 200, 100))
                uploaded_file = io.BytesIO()
                sample_img.save(uploaded_file, format='PNG')
                uploaded_file.seek(0)
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                # Image info
                st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Format:** {image.format if hasattr(image, 'format') else 'N/A'}")
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            if uploaded_file is not None:
                if st.button("🚀 Classify Image", type="primary", key="classify_single"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Preprocess
                            inputs = st.session_state.processor(
                                images=image,
                                return_tensors="pt"
                            ).to(st.session_state.device)
                            
                            # Predict
                            with torch.no_grad():
                                outputs = st.session_state.model(**inputs)
                                logits = outputs.logits
                                probs = torch.nn.functional.softmax(logits, dim=-1)
                            
                            # Get top predictions
                            top_probs, top_indices = torch.topk(probs[0], top_k)
                            
                            # Prepare results
                            results = []
                            for prob, idx in zip(top_probs, top_indices):
                                prob_percent = prob.item() * 100
                                if prob_percent >= confidence_threshold:
                                    class_name = st.session_state.class_names[idx.item()]
                                    results.append({
                                        'class': class_name,
                                        'confidence': prob_percent,
                                        'index': idx.item()
                                    })
                            
                            if results:
                                # Top prediction
                                top_result = results[0]
                                
                                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                                st.markdown(f"### 🏆 Top Prediction")
                                st.markdown(f"## **{top_result['class'].replace('_', ' ').title()}**")
                                st.progress(top_result['confidence'] / 100)
                                st.markdown(f"### {top_result['confidence']:.2f}% Confidence")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # All predictions
                                st.markdown("---")
                                st.markdown("### 📋 All Predictions")
                                
                                # Create dataframe
                                df_results = pd.DataFrame(results)
                                df_results['confidence'] = df_results['confidence'].round(2)
                                df_results['class'] = df_results['class'].apply(lambda x: x.replace('_', ' ').title())
                                
                                # Display as table
                                st.dataframe(
                                    df_results[['class', 'confidence']].rename(
                                        columns={'class': 'Food Item', 'confidence': 'Confidence (%)'}
                                    ),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Visualize as bar chart
                                fig = px.bar(
                                    df_results,
                                    x='confidence',
                                    y='class',
                                    orientation='h',
                                    title='Confidence Scores',
                                    labels={'confidence': 'Confidence (%)', 'class': 'Food Item'},
                                    color='confidence',
                                    color_continuous_scale='RdYlGn'
                                )
                                fig.update_layout(showlegend=False, height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Save to history
                                st.session_state.predictions_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'image_size': image.size,
                                    'top_prediction': top_result['class'],
                                    'top_confidence': top_result['confidence'],
                                    'all_predictions': results
                                })
                                
                            else:
                                st.warning("⚠️ No predictions above the confidence threshold")
                        
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
            else:
                st.info("👆 Upload an image to see predictions")
    
    # Tab 2: Batch Upload
    with tab2:
        st.header("Batch Image Classification")
        
        uploaded_files = st.file_uploader(
            "Upload multiple food images...",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} images uploaded**")
            
            if st.button("🚀 Classify All Images", type="primary", key="classify_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}...")
                    
                    try:
                        image = Image.open(uploaded_file).convert('RGB')
                        
                        # Preprocess
                        inputs = st.session_state.processor(
                            images=image,
                            return_tensors="pt"
                        ).to(st.session_state.device)
                        
                        # Predict
                        with torch.no_grad():
                            outputs = st.session_state.model(**inputs)
                            logits = outputs.logits
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                        
                        # Get top prediction
                        top_prob, top_idx = torch.max(probs[0], dim=0)
                        
                        batch_results.append({
                            'Image': uploaded_file.name,
                            'Prediction': st.session_state.class_names[top_idx.item()].replace('_', ' ').title(),
                            'Confidence (%)': round(top_prob.item() * 100, 2)
                        })
                        
                    except Exception as e:
                        batch_results.append({
                            'Image': uploaded_file.name,
                            'Prediction': f'Error: {str(e)}',
                            'Confidence (%)': 0
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing complete!")
                
                # Display results
                st.subheader("📊 Batch Results")
                df_batch = pd.DataFrame(batch_results)
                st.dataframe(df_batch, use_container_width=True, hide_index=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(batch_results))
                with col2:
                    avg_conf = df_batch['Confidence (%)'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.2f}%")
                with col3:
                    high_conf = len(df_batch[df_batch['Confidence (%)'] >= 80])
                    st.metric("High Confidence (>80%)", high_conf)
                
                # Download results
                csv = df_batch.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "batch_predictions.csv",
                    "text/csv",
                    key='download-batch-csv'
                )
                
                # Visualize distribution
                st.subheader("📈 Prediction Distribution")
                pred_counts = df_batch['Prediction'].value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title='Food Categories Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: History
    with tab3:
        st.header("Prediction History")
        
        if st.session_state.predictions_history:
            st.write(f"**Total Predictions:** {len(st.session_state.predictions_history)}")
            
            # Display history
            for idx, entry in enumerate(reversed(st.session_state.predictions_history)):
                with st.expander(f"🕐 {entry['timestamp']} - {entry['top_prediction'].replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Image Info:**")
                        st.write(f"- Size: {entry['image_size'][0]} x {entry['image_size'][1]} px")
                        st.write(f"- Timestamp: {entry['timestamp']}")
                    
                    with col2:
                        st.write("**Top Prediction:**")
                        st.write(f"- **{entry['top_prediction'].replace('_', ' ').title()}**")
                        st.write(f"- Confidence: **{entry['top_confidence']:.2f}%**")
                    
                    st.write("**All Predictions:**")
                    pred_df = pd.DataFrame(entry['all_predictions'])
                    pred_df['class'] = pred_df['class'].apply(lambda x: x.replace('_', ' ').title())
                    pred_df['confidence'] = pred_df['confidence'].round(2)
                    st.dataframe(
                        pred_df[['class', 'confidence']].rename(
                            columns={'class': 'Food Item', 'confidence': 'Confidence (%)'}
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Download history
            if st.button("📥 Download Complete History (JSON)"):
                json_str = json.dumps(st.session_state.predictions_history, indent=2)
                st.download_button(
                    "Download",
                    json_str,
                    "prediction_history.json",
                    "application/json"
                )
        else:
            st.info("📭 No prediction history yet. Classify some images to see them here!")
    
    # Tab 4: Analytics
    with tab4:
        st.header("Analytics Dashboard")
        
        if st.session_state.predictions_history:
            # Extract data
            all_predictions = []
            all_confidences = []
            timestamps = []
            
            for entry in st.session_state.predictions_history:
                all_predictions.append(entry['top_prediction'])
                all_confidences.append(entry['top_confidence'])
                timestamps.append(entry['timestamp'])
            
            # Summary metrics
            st.subheader("📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Total Predictions", len(all_predictions))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                avg_conf = np.mean(all_confidences)
                st.metric("Avg Confidence", f"{avg_conf:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                max_conf = np.max(all_confidences)
                st.metric("Max Confidence", f"{max_conf:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                unique_classes = len(set(all_predictions))
                st.metric("Unique Classes", unique_classes)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Most Predicted Foods")
                pred_counts = pd.Series(all_predictions).value_counts().head(10)
                fig = px.bar(
                    x=pred_counts.values,
                    y=[p.replace('_', ' ').title() for p in pred_counts.index],
                    orientation='h',
                    labels={'x': 'Count', 'y': 'Food Item'},
                    color=pred_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Confidence Distribution")
                fig = px.histogram(
                    all_confidences,
                    nbins=20,
                    labels={'value': 'Confidence (%)', 'count': 'Frequency'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence over time
            st.subheader("📉 Confidence Over Time")
            time_df = pd.DataFrame({
                'Timestamp': timestamps,
                'Confidence': all_confidences
            })
            fig = px.line(
                time_df,
                x='Timestamp',
                y='Confidence',
                markers=True,
                labels={'Confidence': 'Confidence (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("📊 No data available yet. Start classifying images to see analytics!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🍔 ViT Food Classifier | Powered by Vision Transformer | Built with Streamlit</p>
    <p style='font-size: 0.8rem;'>Using Your Custom Fine-tuned Model</p>
</div>
""", unsafe_allow_html=True)
