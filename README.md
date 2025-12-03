# 🍏 Food Nutrition Intelligence
### *Nutrition Analytics, Modeling, Clustering & Recommendation Platform*

🌐 **Live Demo:**  
https://food-nutrition-intelligence.streamlit.app/

📊 **Dataset (Kaggle):**  
https://www.kaggle.com/datasets/henryshan/food-nutrition-dataset

🎥 **Demo Video:**  
Included in repository (`Demo-Video.mp4`)

Built by **Özge Güneş**

---

# 📌 Overview 

**Food Nutrition Intelligence** is a modern, interactive nutrition analytics platform that analyzes nutrient profiles of **150+ everyday foods** and provides:

- Calorie prediction with a scientifically grounded model  
- Nutrient-based clustering and PCA-powered 2D Food Map  
- Intelligent food similarity & recommendation system  
- Smart Picks (high-protein, low-calorie, vitamin-rich lists)  
- TR/EN bilingual Streamlit interface  
- Modern UI with dark/light theme  

---

# 📚 Scientific Foundation 

This work is aligned with the findings of the paper:  

**Rüede et al. (2020)**  
*Multi-Task Learning for Calorie Prediction on a Novel Large-Scale Recipe Dataset Enriched with Nutritional Information.*

Key insights relevant to this project:

- **Macronutrients are the strongest predictors of calorie content.**  
- **Single-task (kcal-only) models naturally achieve limited accuracy** → approx. **R² ≈ 0.30–0.40**  
- **Multi-task models** or ingredient-level data improve accuracy  
- Nutrient profiles form a **low-dimensional latent structure**, suitable for PCA

➡️ This project's Ridge Regression model produces **CV R² ≈ 0.36**,  
which is **exactly in the scientific accuracy range** reported in the literature.

---

# 🧠 Key Features 

### ✔ 1. Calorie Prediction (Ridge Regression)
Predicts calories using protein, carbohydrates, and fat.  
Aligned with scientific findings on macro–calorie correlation.

### ✔ 2. Nutrient-Based Clustering (K-Means + PCA)
- PCA explains **99.6%** of variance → nutrient data is strongly low-dimensional  
- Visual 2D “Food Map” showing natural nutrient clusters  
- Fruits, bakery items, meats, and snacks cluster intuitively

### ✔ 3. Recommendation Engine (Cosine Similarity)
Suggests nutritionally similar food items.  
Example: replaces high-fat items with lower-calorie alternatives.

### ✔ 4. Smart Picks
Auto-generated lists for:
- High protein  
- Low calorie  
- High vitamin C  
- High iron  

### ✔ 5. Modern UI  
- TR/EN bilingual  
- Clean layout  
- Dark & light themes  

---

# 🏗 Project Architecture 

Food Nutrition Intelligence  
│  
├── Data Layer  
│   ├── USDA-based Kaggle dataset  
│   ├── Cleaning & normalization  
│  
├── Machine Learning  
│   ├── Calorie Model (Ridge Regression)  
│   ├── PCA (2D reduction)  
│   ├── K-Means clustering  
│   └── Cosine similarity engine  
│  
├── Visualization  
│   ├── Plotly interactive charts  
│   ├── Food Map  
│   └── Radar comparison charts  
│  
└── Streamlit UI  
    ├── Explorer  
    ├── Compare  
    ├── ML Lab  
    └── Recommendation  

---

# 🖼 Screenshots 

`/images/`:

- overview.png  
- explorer.png  
- compare.png  
- ml_lab.png  
- recommend.png  

---

# 📊 Results 
### Calorie Model  
- CV R²: **0.36**  
- Matches scientific expectation (R² ≈ 0.30–0.40)

### PCA  
- Explained variance: **99.6%**  
→ Nutrient data clearly low-dimensional

### Clustering  
- Meaningful groupings based on nutrient similarity

### Recommendations  
- High-quality similarity matches  
- Effective for alternative choices  

---

# 💡 Use Cases 

- Diet planning  
- Food comparison  
- Recipe development  
- Healthy alternative discovery  
- Nutrition education  
- FMCG & food analytics  

---

# 🛠 Tech Stack 

- Python  
- Streamlit  
- Pandas  
- NumPy  
- scikit-learn  
- Plotly  
- PCA / K-Means  
- Cosine Similarity  

---

# ⚙ Installation 

pip install -r requirements.txt  
streamlit run app.py  

---

# 👤 Author 

**Özge Güneş**

---

# 📚 References

Rüede, R., Heusser, V., Frank, L., Roitberg, A., Haurilet, M., & Stiefelhagen, R. (2020).
Multi-Task Learning for Calorie Prediction on a Novel Large-Scale Recipe Dataset Enriched with Nutritional Information.
arXiv preprint arXiv:2011.01082.

---

