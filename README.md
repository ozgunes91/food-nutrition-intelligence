# 🍏 Food Nutrition Intelligence
### *AI-Powered Nutrition Analytics, Modeling, Clustering & Recommendation Platform*

🌐 **Live Demo:**  
https://food-nutrition-intelligence.streamlit.app/

📊 **Dataset (Kaggle):**  
https://www.kaggle.com/datasets/henryshan/food-nutrition-dataset

🎥 **Demo Video:**  
Included in repository (`Demo-Video.mp4`)
---

## 🇺🇸 English Overview

### 📘 Project Summary  
**Food Nutrition Intelligence** is a complete analytics + machine learning dashboard designed to explore, model, cluster, and recommend foods based on nutritional data.  
Built with **Streamlit**, **Python**, and **scikit-learn**, this platform analyzes **200+ everyday foods** sourced from USDA FoodData Central.

Features include:
- Interactive nutrition dashboard  
- Calorie prediction ML model  
- K-Means clustering + PCA food mapping  
- Nutrition-based recommendation system  
- Food comparison radar charts  
- Smart Picks (high-protein, low-calorie, high-vitamin-C, etc.)  
- TR/EN bilingual interface  
- Light/Dark Palantir-style UI  

---

## 🎯 Features

### ✔ 1. Interactive Dashboard  
- Explore calories, protein, carbs, fat, iron, vitamin C  
- Filter foods by category or nutrient ranges  
- Macro distribution by category  
- Calories vs protein scatter analysis  

### ✔ 2. Calorie Prediction Model  
- Ridge Regression with cross-validation  
- Predict calories using:  
  **Protein + Carbs + Fat**  
- Overfitting prevented via regularization  
- CV R² score displayed  

### ✔ 3. Unsupervised Learning — Clustering  
- K-Means (2–8 clusters)  
- PCA-based 2D “Food Map”  
- Hoverable nutrient details  

### ✔ 4. Recommendation System  
- Cosine Similarity-based  
- Discover similar foods instantly  
- Bar chart similarity scores  

### ✔ 5. Food Comparison Tool  
- Compare up to 4 foods  
- Radar chart visualization  
- Normalized macro/micro comparison (0–100)  

### ✔ 6. Smart Picks  
- High Protein & Low Fat  
- Low Calorie  
- Iron-Rich  
- Vitamin C Bombs  

---

## 📊 Dataset

**Source:** USDA FoodData Central  
**Rows:** 205 foods  
**Columns:** 9 nutritional features  

| Column | Description |
|--------|-------------|
| food_name | Food item name |
| category | Food category |
| calories | kcal per 100g |
| protein | g |
| carbs | g |
| fat | g |
| iron | mg |
| vitamin_c | mg |
| health_score | Custom health index (0–100) |

Dataset file included:  
```
Food_Nutrition_Dataset.csv
```

---

## 🏗 Repository Structure

```
food-nutrition-intelligence/
│
├── app.py                         # Streamlit application
├── Food_Nutrition_Dataset.csv     # Dataset
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
└── images
    -overview.png
    -explorer.png
    -compare.png
    -ml_lab.png
    -smart_picks.png
    -recommender.png
└── media
    -demo_video.mp4
   
```

---

## 🚀 Deployment (Streamlit Cloud)

1. Push repository to GitHub  
2. Visit: https://share.streamlit.io  
3. Click **Deploy App**  
4. Select your repo → choose **app.py**  
5. Done — your dashboard is publicly live  

---

## 🛠 Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
# 📚 Scientific Foundation (EN)

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

# 🧠 Key Features (EN)

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


# 📊 Results (EN)

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

# 🇹🇷 Türkçe Açıklama

## 📘 Proje Özeti  
**Food Nutrition Intelligence**, 200’ün üzerinde günlük gıdanın besin értlerini analiz etmek, makine öğrenmesi ile kalori tahmini yapmak, kümeler oluşturmak ve benzer gıdaları önermek için geliştirilmiş kapsamlı bir AI destekli dashboard uygulamasıdır.

Uygulama:
- Streamlit  
- Python  
- scikit-learn  
- Plotly  
ile geliştirilmiştir ve kurumsal Palantir tarzı bir tasarıma sahiptir.

---

## 🎯 Özellikler

### ✔ 1. Etkileşimli Dashboard  
- Kalori, protein, karbonhidrat, yağ, demir, C vitamini değerleri  
- Kategori ve besin filtreleme  
- Makro dağılım grafikleri  
- Kalori–protein ilişkisi  

### ✔ 2. Kalori Tahmin Modeli  
- Ridge Regression  
- Düzenlileştirme ile overfit engellenmiş  
- Protein + Karbonhidrat + Yağ → Kalori tahmini  
- 5-katlı CV sonucu gösterilir  

### ✔ 3. K-Means Kümeleme + PCA  
- 2–8 küme seçimi  
- 2D “Food Map”  
- Üzerine gelince detayları gösterir  

### ✔ 4. Benzer Gıda Tavsiye Sistemi  
- Cosine similarity  
- En benzer gıdaları listeler  
- Bar grafik ile puanlar  

### ✔ 5. Gıda Karşılaştırma  
- En fazla 4 gıda  
- Radar grafik  
- Normalize 0–100 karşılaştırma  

### ✔ 6. Akıllı Seçimler  
- Yüksek protein  
- Düşük kalori  
- Demirden zengin  
- C vitamini yüksek  

---

## 📊 Veri Seti

| Kolon | Açıklama |
|--------|----------|
| food_name | Gıda adı |
| category | Gıda kategorisi |
| calories | 100g için kalori |
| protein | g |
| carbs | g |
| fat | g |
| iron | mg |
| vitamin_c | mg |
| health_score | 0–100 arası sağlık skoru |

---

## 🏗 Depo Yapısı


```
food-nutrition-intelligence/
│
├── app.py                         # Streamlit application
├── Food_Nutrition_Dataset.csv     # Dataset
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
└── images
    -overview.png
    -explorer.png
    -compare.png
    -ml_lab.png
    -smart_picks.png
    -recommender.png
└── media
    -demo_video.mp4
   
```

---

## 🚀 Dağıtım (Streamlit Cloud)

1. Projeyi GitHub’a yükle  
2. https://share.streamlit.io adresine gir  
3. “Deploy App”  
4. app.py dosyasını seç  
5. Uygulama internette herkese açık hale gelir  

---

## 🛠 Lokal Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

---
# 📚 Bilimsel Temel (TR)

Bu çalışma, aşağıdaki makalenin bulgularıyla uyumludur:

**Rüede ve ark. (2020)**  
*Multi-Task Learning for Calorie Prediction on a Novel Large-Scale Recipe Dataset Enriched with Nutritional Information.*

Bu projeyle doğrudan ilişkili bulgular:

- **Makro besinler kalori içeriğinin en güçlü belirleyicisidir.**  
- **Tek görevli (yalnızca kalori tahmini) modellerde doğruluk sınırlıdır** → **R² ≈ %30–40**  
- **Multi-task modeller** ve ingredient-level veri doğruluğu artırır  
- Besin profilleri **düşük boyutlu bir yapıya** sahiptir (PCA için uygundur)

➡️ Bu projedeki Ridge Regression modeli **CV R² ≈ %36** üretmiştir  
ve literatürdeki beklenti aralığıyla **birebir uyumludur**.

---
# 🧠 Temel Özellikler (TR)

### ✔ 1. Kalori Tahmini (Ridge Regression)
Protein, karbonhidrat ve yağ değerlerini kullanarak kalori tahmini yapar.  
Makro besin – kalori ilişkisine dair bilimsel bulgularla uyumludur.

### ✔ 2. Besin Tabanlı Kümeleme (K-Means + PCA)
- PCA toplam varyansın **%99.6’sını** açıklar → besin verisi güçlü şekilde düşük boyutludur  
- 2D “Besin Haritası” doğal kümeleri görselleştirir  
- Meyveler, etler, unlu mamuller vb. mantıklı şekilde kümelenir

### ✔ 3. Öneri Motoru (Cosine Similarity)
Besin profiline benzer yiyecekleri bulur.  
Örn: yüksek yağlı ürün yerine daha düşük kalorili alternatifler.

### ✔ 4. Smart Picks  
Otomatik listeler:

- Yüksek protein  
- Düşük kalori  
- Yüksek C vitamini  
- Yüksek demir  

### ✔ 5. Modern Arayüz  
- TR/EN çift dil  
- Temiz düzen  
- Karanlık / aydınlık tema  

---
# 📊 Results (EN)

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

## ✨ Author  
**Özge Güneş**  
AI & Data Science Portfolio  

---

# 📚 References/Referans

Rüede, R., Heusser, V., Frank, L., Roitberg, A., Haurilet, M., & Stiefelhagen, R. (2020).
Multi-Task Learning for Calorie Prediction on a Novel Large-Scale Recipe Dataset Enriched with Nutritional Information.
arXiv preprint arXiv:2011.01082.

---
