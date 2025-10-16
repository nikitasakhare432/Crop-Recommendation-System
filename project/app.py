import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8fff8;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .crop-result {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #666;
        font-size: 14px;
    }
    h1 {
        color: #2E7D32;
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        color: #388E3C;
    }
    .stNumberInput>div>div>input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'crop_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_dataset():
    """Load the dataset for visualizations"""
    csv_path = 'Crop_recommendation.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def main():
    st.markdown("<h1>🌾 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 18px;'>Predict the best crop based on soil nutrients and weather conditions 🌦️</p>", unsafe_allow_html=True)

    model = load_model()
    df = load_dataset()

    if model is None:
        st.error("⚠️ Model not found! Please run 'python train_model.py' first to train the model.")
        st.info("📥 Download the dataset from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        return

    st.markdown("---")

    st.subheader("🌱 Enter Soil and Weather Parameters")

    col1, col2 = st.columns(2)

    with col1:
        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0,
            max_value=150,
            value=50,
            help="Nitrogen content ratio in soil"
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0,
            max_value=150,
            value=50,
            help="Phosphorus content ratio in soil"
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0,
            max_value=210,
            value=50,
            help="Potassium content ratio in soil"
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            step=0.1,
            help="Temperature in Celsius"
        )

    with col2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1,
            help="Relative humidity in percentage"
        )

        ph = st.number_input(
            "pH Value",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            help="pH value of the soil"
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=0.1,
            help="Rainfall in mm"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        predict_button = st.button("🌱 Recommend Crop", use_container_width=True)

    if predict_button:
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        prediction = model.predict(input_data)[0]

        st.markdown(f"""
            <div class="crop-result">
                🎉 Recommended Crop: <span style="font-size: 32px;">{prediction.upper()}</span> 🌾
            </div>
        """, unsafe_allow_html=True)

        st.success(f"✅ Based on the provided conditions, **{prediction}** is the best crop to cultivate!")

    st.markdown("---")

    if df is not None:
        with st.sidebar:
            st.markdown("## 📈 Data Insights")
            st.markdown("These visuals help understand crop patterns under different conditions.")

            show_insights = st.checkbox("Show Visualizations", value=True)

        if show_insights:
            st.markdown("## 📊 Dataset Visualizations")

            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Rainfall Distribution",
                "🌡️ Temperature Analysis",
                "🔬 pH vs Humidity",
                "📈 Nutrient Analysis"
            ])

            with tab1:
                st.subheader("Rainfall Distribution by Crop")
                fig, ax = plt.subplots(figsize=(12, 6))
                df.boxplot(column='rainfall', by='label', ax=ax, patch_artist=True)
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Crop Type')
                plt.ylabel('Rainfall (mm)')
                plt.title('Rainfall Distribution by Crop Type')
                plt.suptitle('')
                ax.set_facecolor('#f8fff8')
                fig.patch.set_facecolor('#f8fff8')
                st.pyplot(fig)
                plt.close()

            with tab2:
                st.subheader("Average Temperature per Crop")
                fig, ax = plt.subplots(figsize=(12, 6))
                avg_temp = df.groupby('label')['temperature'].mean().sort_values(ascending=False)
                bars = ax.bar(range(len(avg_temp)), avg_temp.values, color='#4CAF50')
                ax.set_xticks(range(len(avg_temp)))
                ax.set_xticklabels(avg_temp.index, rotation=45, ha='right')
                ax.set_xlabel('Crop Type')
                ax.set_ylabel('Average Temperature (°C)')
                ax.set_title('Average Temperature Requirements by Crop')
                ax.set_facecolor('#f8fff8')
                fig.patch.set_facecolor('#f8fff8')
                st.pyplot(fig)
                plt.close()

            with tab3:
                st.subheader("pH vs Humidity (Colored by Crop)")
                fig, ax = plt.subplots(figsize=(12, 8))
                crops = df['label'].unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(crops)))

                for crop, color in zip(crops, colors):
                    crop_data = df[df['label'] == crop]
                    ax.scatter(crop_data['ph'], crop_data['humidity'],
                             label=crop, alpha=0.6, s=50, color=color)

                ax.set_xlabel('pH Value')
                ax.set_ylabel('Humidity (%)')
                ax.set_title('pH vs Humidity Relationship by Crop Type')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8fff8')
                fig.patch.set_facecolor('#f8fff8')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with tab4:
                st.subheader("NPK Nutrient Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                nutrients = df.groupby('label')[['N', 'P', 'K']].mean()
                nutrients.plot(kind='bar', ax=ax, color=['#8BC34A', '#4CAF50', '#2E7D32'])
                ax.set_xlabel('Crop Type')
                ax.set_ylabel('Average Nutrient Level')
                ax.set_title('Average NPK Requirements by Crop')
                ax.legend(['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'])
                plt.xticks(rotation=45, ha='right')
                ax.set_facecolor('#f8fff8')
                fig.patch.set_facecolor('#f8fff8')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.info("💡 **Note:** These visualizations are generated from the training dataset to help understand optimal growing conditions for different crops.")

    else:
        st.warning("⚠️ Please upload 'Crop_recommendation.csv' to view insights.")

    st.markdown("""
        <div class="footer">
            <p>🌾 <strong>Developed by Nikita Sakhare</strong> | Powered by Machine Learning 🤖</p>
            <p style="font-size: 12px; margin-top: 10px;">
                Using RandomForestClassifier with 100 estimators | Trained on Kaggle Crop Recommendation Dataset
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
