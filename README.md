# 🍏 Food Nutrition Intelligence
### **AI-Powered Nutrition Intelligence Platform**  
*A modern, interactive platform for analyzing, modeling, clustering, and recommending foods using nutrition data.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ozgunes91/food-nutrition-intelligence/main/images/overview.png" width="85%">
</p>

---

# 🌐 Live Demo  
👉 https://food-nutrition-intelligence.streamlit.app/

# 📊 Dataset  
👉 https://www.kaggle.com/datasets/henryshan/food-nutrition-dataset

# 🎥 Demo Video  
Located in `media/Demo-Video.mp4`

---

📊 Veri Seti:  
https://www.kaggle.com/datasets/henryshan/food-nutrition-dataset

🎥 Demo Videosu:  
`media/Demo-Video.mp4`

---

## 🚀 Project Overview
**Food Nutrition Intelligence** is a machine learning platform designed to analyze the nutrient profiles of **200+ foods** and provide meaningful nutritional insights.

The system performs:
- **Calorie prediction** using Ridge Regression  
- **Clustering** based on nutrient similarity  
- **Similarity-based food recommendations**  
- **2D PCA-based Nutrient Profile Map**  
- **Interactive visualization** via Streamlit  
- **Bilingual UI (EN/TR)**  

This project was developed as part of continuous data science practice, with an emphasis on scientific transparency and practical ML application.

## 📌 Scientific Foundation
This project is inspired by:

**Rüede et al. (2020)**  
*Multi-Task Learning for Calorie Prediction on a Novel Large-Scale Recipe Dataset Enriched with Nutritional Information.*

Key insights from the literature:
- Macronutrients are strong—but incomplete—predictors of calorie content.  
- Models trained only on macronutrients naturally show **limited predictive power**, because they lack critical information such as:
  - Cooking method  
  - Moisture loss  
  - Fat absorption  
  - Processing level  
  - Ingredient composition  
- Multi-task and ingredient-level models significantly improve accuracy.  
- Nutrition data often exhibits a **low-dimensional structure**, making PCA an effective tool for exploration.

In this project, a Ridge Regression model trained solely on the three macronutrients (fat, protein, carbohydrates) achieved a **cross-validated R² of 0.36**.  
This modest performance is consistent with the **expected behavior of macro-only models**, whose predictive capacity is inherently limited by missing preparation-related features (see e.g., Rüede et al., 2020).

## 📈 Results
### 🔹 Calorie Model
- **CV R²:** 0.36  
- **Interpretation:**  
  The model performs exactly as expected for macro-only calorie prediction, where limited feature diversity caps predictive power.

### 🔹 Clustering
- K-Means clustering  
- Scaled nutrient profiles  
- Visual cluster boundaries displayed in the Streamlit app  

### 🔹 Similarity Engine
- Cosine similarity matrix  
- “Top similar foods” recommendation tool  

### 🔹 PCA Nutrient Map
- 2 principal components capture **99.6%** of total variance  
- 2D interactive nutrient landscape via Plotly  

## 🧭 Features & Capabilities
- ✔ Ridge Regression with sklearn pipeline  
- ✔ StandardScaler preprocessing  
- ✔ K-Means clustering & silhouette-based tuning  
- ✔ Cosine similarity recommendations  
- ✔ PCA dimensionality reduction  
- ✔ Streamlit UI with bilingual support  
- ✔ Clean modular code structure  

## 🌐 Live Demo
👉 https://lnkd.in/diGSfhrY

## 📁 GitHub Repository
👉 https://github.com/ozgunes91/food-nutrition-intelligence

## 🗂️ Repository Structure
```
food-nutrition-intelligence/
│
├── app.py
├── Food_Nutrition_Dataset.csv           
├── requirements.txt
├── README.md
├── images
├── media

```

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Plotly**
- **Streamlit**
- **Cosine similarity**
- **PCA**

## 📚 Reference
Rüede, R. et al. (2020).  
*Multi-Task Learning for Calorie Prediction from Food Images.*  
arXiv:2011.01082.

## ✨ Feedback
This project was created for learning, experimentation, and portfolio development.  
Feedback and suggestions are always welcome!
