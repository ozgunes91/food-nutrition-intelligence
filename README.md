# 🍏 Food Nutrition Intelligence  
### AI-Powered Nutrition Dashboard with ML, Clustering & Recommendations  
### (TR/EN Bilingual README)

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
├── app.py
├── Food_Nutrition_Dataset.csv
├── requirements.txt
└── README.md
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

## ✨ Author  
**Özge Güneş**  
AI & Data Science Portfolio  
