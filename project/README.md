# 🌾 Smart Crop Recommendation System

An intelligent web application that predicts the best crop to cultivate based on soil nutrients and weather conditions using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

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

## 🚀 Installation & Setup

### Prerequisites:
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project
```bash
git clone <repository-url>
cd smart-crop-recommendation
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
1. Visit: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Download `Crop_recommendation.csv`
3. Place it in the project root directory

### Step 4: Train the Model
```bash
python train_model.py
```

This will:
- Load and process the dataset
- Train the RandomForestClassifier
- Save the trained model as `crop_model.pkl`
- Display accuracy metrics and feature importance

### Step 5: Run the Application
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

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

## 🌐 Deployment

### Option 1: Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Visit: https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Important:** Upload `Crop_recommendation.csv` and `crop_model.pkl` to your repository before deploying.

### Option 2: Bolt.new

1. Upload all project files to Bolt.new
2. Ensure `crop_model.pkl` is present
3. The app will auto-deploy and be accessible via unique URL

### Option 3: Local Deployment

```bash
streamlit run app.py --server.port 8501
```

---

## 🎨 Design Features

### Color Palette:
- Primary Green: `#4CAF50`
- Light Green: `#8BC34A`
- Dark Green: `#2E7D32`
- Background: `#f8fff8` (Light mint)

### UI Elements:
- ✅ Rounded buttons with hover effects
- ✅ Two-column input layout
- ✅ Gradient result cards
- ✅ Responsive design
- ✅ Interactive visualizations
- ✅ Clean typography

### Icons Used:
- 🌾 Wheat (crop/agriculture)
- 🌱 Seedling (growth/recommendation)
- 🌦️ Weather (conditions)
- 📊 Charts (data insights)
- 🤖 Robot (machine learning)

---

## 🧠 Model Performance

### Training Metrics:
- **Algorithm**: RandomForestClassifier
- **Estimators**: 100 trees
- **Max Depth**: 20
- **Train/Test Split**: 80/20
- **Accuracy**: ~99%
- **Cross-validation**: Stratified sampling

### Feature Importance:
The model analyzes all 7 features, with typical importance ranking:
1. Rainfall
2. Potassium (K)
3. Phosphorus (P)
4. Temperature
5. Humidity
6. Nitrogen (N)
7. pH

---

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit 1.28.0**: Web framework
- **Scikit-learn 1.3.1**: Machine learning
- **Pandas 2.1.1**: Data manipulation
- **NumPy 1.26.0**: Numerical computing
- **Matplotlib 3.8.0**: Data visualization
- **Seaborn 0.13.0**: Statistical visualizations

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement:
- Add more crop types
- Implement real-time weather API integration
- Add location-based recommendations
- Include crop disease prediction
- Mobile app version

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Developer

**Developed by Priyanka Sakhare**
Powered by Machine Learning 🤖

---

## 🐛 Troubleshooting

### Issue: Model not found
**Solution:** Run `python train_model.py` first to train and save the model.

### Issue: Dataset not found
**Solution:** Download `Crop_recommendation.csv` from Kaggle and place it in the project root.

### Issue: Import errors
**Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: Streamlit not starting
**Solution:** Check if port 8501 is available or specify a different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- Dataset provided by Atharva Ingle on Kaggle
- Streamlit for the amazing web framework
- Scikit-learn for machine learning tools
- The open-source community

---

**Made with ❤️ for Farmers and Agriculture Enthusiasts**
