# 🌾 Smart Crop Recommendation System

## 📋 Project Overview

The Smart Crop Recommendation System leverages a **RandomForestClassifier** trained on agricultural data to help farmers and agricultural professionals make data-driven decisions about crop cultivation. The system analyzes soil nutrients (N, P, K), weather conditions (temperature, humidity, rainfall), and soil pH to recommend the most suitable crop.

### Key Features:
- 🤖 **Machine Learning**: RandomForest algorithm with 100 estimators
- 🌐 **Interactive Web Interface**: Built with Streamlit
- 📊 **Data Visualizations**: Comprehensive charts and graphs
- 🎨 **Beautiful UI**: Green-themed, farm-inspired design
- 📈 **High Accuracy**: Achieves 99%+ accuracy on test data
- 🚀 **Easy Deployment**: One-click deployment ready

---

## 📊 Dataset Source

**Kaggle Crop Recommendation Dataset**
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

### Dataset Details:
- **Records**: 2,200 samples
- **Features**: 7 input features
- **Target**: 22 crop types
- **Crops Included**: Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

### Input Features:
1. **N** - Nitrogen content ratio in soil
2. **P** - Phosphorus content ratio in soil
3. **K** - Potassium content ratio in soil
4. **Temperature** - Temperature in Celsius
5. **Humidity** - Relative humidity in percentage
6. **pH** - pH value of the soil
7. **Rainfall** - Rainfall in mm



---

## 💻 Usage Guide

### Making Predictions:

1. **Enter Soil Nutrients:**
   - Nitrogen (N): 0-150
   - Phosphorus (P): 0-150
   - Potassium (K): 0-210

2. **Enter Weather Conditions:**
   - Temperature: 0-50°C
   - Humidity: 0-100%
   - Rainfall: 0-300mm

3. **Enter Soil Properties:**
   - pH Value: 0-14

4. **Click "🌱 Recommend Crop"**

5. **View Results:**
   - Recommended crop displayed prominently
   - Success message with cultivation advice

### Viewing Data Insights:

Navigate through the tabs to explore:
- 📊 **Rainfall Distribution**: Boxplots showing rainfall patterns for each crop
- 🌡️ **Temperature Analysis**: Average temperature requirements
- 🔬 **pH vs Humidity**: Scatter plot relationships
- 📈 **Nutrient Analysis**: NPK requirements comparison

---

## 📝 Example Inputs & Outputs

### Example 1: Rice
**Input:**
- N: 80, P: 40, K: 40
- Temperature: 25°C
- Humidity: 80%
- pH: 6.5
- Rainfall: 200mm

**Output:** ✅ Recommended Crop: **RICE**

### Example 2: Coffee
**Input:**
- N: 100, P: 20, K: 30
- Temperature: 23°C
- Humidity: 60%
- pH: 6.5
- Rainfall: 120mm

**Output:** ✅ Recommended Crop: **COFFEE**

### Example 3: Cotton
**Input:**
- N: 120, P: 40, K: 20
- Temperature: 28°C
- Humidity: 70%
- pH: 7.0
- Rainfall: 80mm

**Output:** ✅ Recommended Crop: **COTTON**

---

## 📁 File Structure

```
smart-crop-recommendation/
│
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── Crop_recommendation.csv         # Dataset (download separately)
├── crop_model.pkl                  # Trained model (generated)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---

