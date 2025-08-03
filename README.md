# 🏠 California House Pricing

A Jupyter/Colab project for analyzing and predicting California housing prices using real-world data. The app visualizes and predicts house prices based on geolocation and socio-economic features.

🔗 **Detailed Report**: [Report Link](https://www.notion.so/California-Housing-Price-Prediction-Report-244428aa03ca80dabfccd353382f3251)
🔗 **Live App**: [Streamlit Deployment](https://california-housing-pricing.streamlit.app/)

---

## 📁 Project Overview

* `preprocess.py`: Contains preprocessing essential functions to be used in both the notebook and app.py.
* `app.py`: Interactive Streamlit web app for predictions.
* `California_House_Pricing.ipynb`: Full pipeline from EDA to model evaluation
* `requirements.txt`: Dependencies
* `README.md`: Project summary

---

## 🔍 Workflow Summary

### 1. 📊 Exploratory Data Analysis (EDA)

* Identified missing data in `total_bedrooms`
* Observed skewed distributions in income and housing features
* Visualized geospatial relationships (latitude, longitude) with price
* Found strong correlation of **`median_income`** with **`median_house_value`**

### 2. 🧪 Data Splitting

* Used **stratified sampling** based on **median income categories** to ensure representative test set

### 3. 🧹 Preprocessing

* Filled missing values with median
* Feature scaling using `StandardScaler`
* Categorical encoding of `ocean_proximity`
* Added combined attributes like:
  * `rooms_per_household`
  * `bedrooms_per_room`
  * `population_per_household`
  * `location`

### 4. 🤖 Modeling
* Chose **Random Forest Regressor** as the primary model due to its:
    - Resistance to overfitting on high-dimensional data.
    - Ability to model non-linear relationships and feature interactions.
    - Inherent support for feature importance extraction.
* Evaluated using **cross-validation (RMSE)**
* Used **RandomizedSearchCV** to tune `RandomForestRegressor`

### 5. 📈 Evaluation

* Final RMSE reported for tuned Random Forest
* Feature importance plotted and interpreted

---

## 💡 Key App Features

* 🗺️ **Map**: Shows house prices spatially with price indicators
* 📥 **User Input**: Custom location and housing features
* 📊 **Prediction**: Real-time price estimation
* 📉 **Model Insights**: Displays important features and price drivers

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/AhmedQassemDev2004/Calefornia-House-Pricing.git
cd Calefornia-House-Pricing
pip install -r requirements.txt
streamlit run app.py
```