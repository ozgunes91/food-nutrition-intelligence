import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Food Nutrition Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "Food_Nutrition_Dataset.csv"

LANG_MAP = {"Türkçe": "tr", "English": "en"}

TEXT = {
    "app_title": {
        "tr": "Food Nutrition Intelligence – Besin Zekâ Paneli",
        "en": "Food Nutrition Intelligence – Nutrition Intelligence Panel",
    },
    "subtitle": {
        "tr": "150+ günlük gıdanın kalori ve besin değerlerini keşfet, modelle, kümele ve öner.",
        "en": "Explore, model, cluster and recommend using nutrition profiles of 150+ everyday foods.",
    },
    "sidebar_language": {"tr": "Dil / Language", "en": "Language"},
    "sidebar_theme": {"tr": "Tema", "en": "Theme"},
    "theme_dark": {"tr": "Karanlık", "en": "Dark"},
    "theme_light": {"tr": "Aydınlık", "en": "Light"},
    "sidebar_select_all": {"tr": "Tümünü Seç / Kaldır", "en": "Select / Deselect All"},
    "sidebar_categories": {"tr": "Kategoriler", "en": "Categories"},
    "sidebar_categories_selected": {"tr": "kategori seçildi", "en": "categories selected"},
    "sidebar_open_category_list": {"tr": "Kategori Listesini Aç", "en": "Open Category List"},
    "sidebar_calorie_range": {"tr": "Kalori aralığı (kcal)", "en": "Calorie range (kcal)"},
    "sidebar_focus": {"tr": "Besin odağı", "en": "Nutrient focus"},
    "focus_all": {"tr": "Hepsi (Genel)", "en": "All (General)"},
    "focus_high_protein": {"tr": "Yüksek Protein", "en": "High Protein"},
    "focus_low_carb": {"tr": "Düşük Karbonhidrat", "en": "Low Carb"},
    "focus_low_fat": {"tr": "Düşük Yağ", "en": "Low Fat"},
    "overview_tab": {"tr": "Genel Bakış", "en": "Overview"},
    "explorer_tab": {"tr": "Keşif", "en": "Explorer"},
    "compare_tab": {"tr": "Karşılaştır", "en": "Compare"},
    "ml_tab": {"tr": "ML Lab (Tahmin + Kümeleme)", "en": "ML Lab (Prediction + Clustering)"},
    "recommender_tab": {"tr": "Tavsiye Sistemi", "en": "Recommender"},
    "smartpicks_tab": {"tr": "Akıllı Seçimler", "en": "Smart Picks"},
    "kpi_total_foods": {"tr": "Toplam Gıda", "en": "Total Foods"},
    "kpi_total_categories": {"tr": "Kategori Sayısı", "en": "Categories"},
    "kpi_median_calories": {"tr": "Medyan Kalori", "en": "Median Calories"},
    "kpi_top_health": {"tr": "En Yüksek Sağlık Skoru", "en": "Top Health Score"},
    "sidebar_control": {"tr": "Kontrol Paneli", "en": "Control Panel"},
    "compare_detail": {"tr": "Detay", "en": "Detail"},
    "recom_howmany": {"tr": "Kaç benzer gıda listelensin?", "en": "How many similar foods?"},
    "recom_button": {"tr": "Benzer Gıdaları Bul", "en": "Find Similar Foods"},
    "ml_predict_button": {"tr": "Kalori Tahmin Et", "en": "Predict Calories"},

    "section_category_macros": {
        "tr": "Kategori Bazlı Ortalama Makro Dağılımı",
        "en": "Average Macro Distribution by Category",
    },
    "section_calorie_protein_scatter": {
        "tr": "Kalori vs Protein (Kategori Bazlı)",
        "en": "Calories vs Protein (by Category)",
    },
    "explorer_title": {
        "tr": "Gıda Keşfi ve Dağılım Analizi",
        "en": "Food Explorer & Distribution",
    },
    "explorer_description": {
        "tr": "Filtreleri kullanarak gıdaları incele, besin dağılımını görselleştir.",
        "en": "Use filters to explore foods and visualize nutrient distributions.",
    },
    "explorer_select_nutrient": {
        "tr": "Histogram için besin seç",
        "en": "Select nutrient for histogram",
    },
    "compare_title": {"tr": "Gıda Karşılaştırma", "en": "Food Comparison"},
    "compare_instruction": {
        "tr": "En fazla 4 gıda seçip besin profillerini radar grafikte karşılaştır.",
        "en": "Select up to 4 foods to compare their nutrient profiles on a radar chart.",
    },
    "compare_warning": {
        "tr": "Karşılaştırma için en az 2 gıda seç.",
        "en": "Select at least 2 foods to compare.",
    },
    "ml_title": {"tr": "ML Lab: Kalori Tahmini & Kümeleme", "en": "ML Lab: Calorie Prediction & Clustering"},
    "ml_pred_title": {"tr": "Kalori Tahmini (Ridge Regression)", "en": "Calorie Prediction (Ridge Regression)"},
    "ml_pred_desc": {
        "tr": "Protein / Karbonhidrat / Yağ değerlerine göre kaloriyi tahmin eder. Basit, düzenlileştirilmiş (regularized) bir model kullanılır ki overfit olmasın.",
        "en": "Predicts calories from Protein / Carbs / Fat using a simple regularized model to avoid overfitting.",
    },
    "ml_cluster_title": {"tr": "Gıda Kümeleme (K-Means + PCA)", "en": "Food Clustering (K-Means + PCA)"},
    "ml_cluster_desc": {
        "tr": "Besin profillerine göre gıdaları kümeler; PCA ile 2D 'Food Map' oluşturur.",
        "en": "Clusters foods by nutritional profile and creates a 2D 'Food Map' using PCA.",
    },
    "recom_title": {"tr": "Benzer Gıda Önerileri", "en": "Similar Food Recommendations"},
    "recom_desc": {
        "tr": "Seçtiğin bir gıdaya besin profili olarak en çok benzeyen gıdaları listeler.",
        "en": "Finds foods with the most similar nutrient profile to the one you select.",
    },
    "smartpicks_title": {"tr": "Akıllı Seçimler", "en": "Smart Picks"},
    "smartpicks_desc": {
        "tr": "Hazır filtreler ile hızlıca sağlıklı alternatifler bul.",
        "en": "Use pre-defined filters to quickly find healthy options.",
    },
    "smartpicks_mode": {"tr": "Mod seç", "en": "Select mode"},
    "smart_high_protein": {"tr": "Yüksek Protein & Düşük Yağ", "en": "High Protein & Low Fat"},
    "smart_low_calorie": {"tr": "Düşük Kalorili", "en": "Low Calorie"},
    "smart_high_iron": {"tr": "Demirden Zengin", "en": "Iron Rich"},
    "smart_vitc": {"tr": "Vitamin C Yüksek", "en": "High Vitamin C"},
    "table_food": {"tr": "Gıda Listesi", "en": "Food List"},
    "health_score_label": {"tr": "Sağlık Skoru (0–100)", "en": "Health Score (0–100)"},
    "no_results": {"tr": "Seçili filtrelere göre sonuç bulunamadı.", "en": "No results for the selected filters."},
}

NUTR_LABELS = {
    "calories": {"tr": "Kalori (kcal)", "en": "Calories (kcal)"},
    "protein": {"tr": "Protein (g)", "en": "Protein (g)"},
    "carbs": {"tr": "Karbonhidrat (g)", "en": "Carbohydrates (g)"},
    "fat": {"tr": "Yağ (g)", "en": "Fat (g)"},
    "iron": {"tr": "Demir (mg)", "en": "Iron (mg)"},
    "vitamin_c": {"tr": "Vitamin C (mg)", "en": "Vitamin C (mg)"},
    "health_score": {"tr": "Sağlık Skoru", "en": "Health Score"},
}


def t(key: str, lang: str) -> str:
    return TEXT.get(key, {}).get(lang, key)


def nl(col: str, lang: str) -> str:
    return NUTR_LABELS.get(col, {}).get(lang, col)


# =========================================================
# THEME / CSS (Palantir Glass Dark & Light)
# =========================================================
def inject_css(theme: str):
    if theme == "Dark":
        css = """
        <style>
        /* Header tamamen yok */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* Ana arka plan – Palantir Dark Glass */
        body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left,#020617 0,#020617 15%,#020617 40%,#020617 100%) !important;
            color: #E5E7EB !important;
        }

        /* Ana içerik alanı */
        [data-testid="stAppViewContainer"] .main {
            padding-top: 1.2rem;
        }

        /* Sidebar – koyu cam efekti */
        section[data-testid="stSidebar"] {
            background: rgba(15,23,42,0.92) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            border-right: 1px solid rgba(148,163,184,0.18) !important;
            box-shadow: 12px 0 40px rgba(0,0,0,0.5);
            color: #E5E7EB !important;
        }

        /* Sidebar iç yazılar */
        section[data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }

        /* Ana header banner */
        .app-header-banner {
            width: 100%;
            padding: 1.4rem 1.8rem;
            border-radius: 1.1rem;
            background: radial-gradient(circle at top left, rgba(16,24,40,0.96), rgba(15,23,42,0.92));
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(148,163,184,0.45);
            box-shadow: 0 26px 70px rgba(15,23,42,0.9);
            display: flex;
            align-items: center;
            margin-bottom: 1.3rem;
        }

        .app-header-left {
            display: flex;
            align-items: center;
            gap: 1.1rem;
        }

        /* Premium elma logosu – cam chip */
        .app-logo-chip {
            width: 52px;
            height: 52px;
            border-radius: 999px;
            background: conic-gradient(from 160deg,#22c55e,#a3e635,#22c55e,#4ade80);
            box-shadow: 0 0 14px rgba(34,197,94,0.7);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .app-logo-inner {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            background: radial-gradient(circle at top,#042f2e,#022c22);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .app-logo-emoji {
            font-size: 1.9rem;
            filter: drop-shadow(0 0 8px rgba(190,242,100,0.9));
        }

        .app-title {
            font-size: 2.45rem !important;
            font-weight: 900 !important;
            letter-spacing: -0.03em;
            color: #A7F3D0 !important;
        }

        .app-subtitle {
            font-size: 1.05rem !important;
            color: #E5E7EB !important;
            opacity: 0.9;
        }

        /* Başlıklar */
        h1, h2, h3, h4, h5 {
            color: #A7F3D0 !important;
        }
        h2 {
            font-size: 26px !important;
            font-weight: 800 !important;
        }
        h3, h4 {
            font-size: 20px !important;
            font-weight: 700 !important;
        }

        /* Genel metin */
        p, span, label, div, .stMarkdown, .stText, .stCaption {
            color: #E5E7EB !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            padding-bottom: 4px !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #CBD5F5 !important;
            font-weight: 500 !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #A7F3D0 !important;
            font-weight: 700 !important;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #22c55e !important;
        }

        /* KPI grid & kartlar */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 20px;
            width: 100%;
            margin-top: 10px;
            margin-bottom: 6px;
        }
        .kpi-card {
            background: radial-gradient(circle at top left,rgba(15,23,42,0.98),rgba(15,23,42,0.92));
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.38);
            padding: 0.9rem 1.3rem;
            box-shadow: 0 22px 60px rgba(15,23,42,0.95);
        }
        .kpi-card h3 {
            margin: 0;
            font-size: 0.98rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: #E5E7EB !important;
        }
        .kpi-card .value {
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0.25rem 0 0.15rem 0;
            color: #F9FAFB !important;
        }
        .kpi-card p {
            margin: 0;
            font-size: 0.82rem;
            color: #9CA3AF !important;
        }

        /* Butonlar */
        .stButton button {
            background: linear-gradient(135deg,#22c55e,#4ade80) !important;
            color: #022c22 !important;
            font-weight: 800 !important;
            border-radius: 999px !important;
            border: none !important;
            box-shadow: 0 10px 30px rgba(34,197,94,0.45) !important;
            padding: 0.55rem 1.5rem !important;
            transition: transform 0.13s ease-out, box-shadow 0.13s ease-out;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 40px rgba(34,197,94,0.6) !important;
        }

        /* Input & Select – koyu cam */
        input, textarea {
            background: rgba(15,23,42,0.85) !important;
            border-radius: 10px !important;
            border: 1px solid #334155 !important;
            color: #E5E7EB !important;
        }
        input:hover, textarea:hover {
            border-color: #22c55e !important;
        }
        input::placeholder, textarea::placeholder {
            color: #9CA3AF !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] > div {
            background: rgba(15,23,42,0.9) !important;
            border-radius: 10px !important;
            border: 1px solid #334155 !important;
        }
        div[data-baseweb="select"] * {
            color: #E5E7EB !important;
        }
        ul[role="listbox"] {
            background: rgba(15,23,42,0.98) !important;
            border-radius: 12px !important;
            border: 1px solid #334155 !important;
        }
        ul[role="listbox"] li {
            background: transparent !important;
        }
        ul[role="listbox"] li * {
            color: #E5E7EB !important;
        }
        ul[role="listbox"] li:hover {
            background: #22c55e !important;
        }
        ul[role="listbox"] li:hover * {
            color: #020617 !important;
        }

        /* Slider */
        .stSlider > div[data-baseweb="slider"] > div > div {
            background: #1e293b !important;
        }
        .stSlider [role="slider"] {
            background: #22c55e !important;
            box-shadow: 0 0 0 3px rgba(34,197,94,0.35) !important;
        }

        /* plotly title rengi */
        .js-plotly-plot .plotly .gtitle {
            fill: #E5E7EB !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            box-shadow: none !important;
        }

        body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom,#f9fafb,#e5e7eb) !important;
            color: #111827 !important;
        }

        [data-testid="stAppViewContainer"] .main {
            padding-top: 1.2rem;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
            box-shadow: 10px 0 35px rgba(15,23,42,0.08);
            color:#1e293b !important;
        }
        section[data-testid="stSidebar"] * {
            color:#1e293b !important;
        }

        /* Light Glass header */
        .app-header-banner {
            width: 100%;
            padding: 1.1rem 1.6rem;
            border-radius: 1rem;
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(226,232,240,0.9);
            box-shadow: 0 20px 55px rgba(148,163,184,0.45);
            display: flex;
            align-items: center;
            margin-bottom: 1.1rem;
        }

        .app-header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .app-logo-chip {
            width: 52px;
            height: 52px;
            border-radius: 999px;
            background: conic-gradient(from 160deg,#16a34a,#a3e635,#22c55e,#16a34a);
            box-shadow: 0 0 12px rgba(22,163,74,0.55);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .app-logo-inner {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            background: radial-gradient(circle at top,#ecfdf5,#bbf7d0);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .app-logo-emoji {
            font-size: 1.8rem;
        }

        .app-title {
            font-size: 2.35rem !important;
            font-weight: 900 !important;
            color: #047857 !important;
        }
        .app-subtitle {
            font-size: 1.02rem !important;
            color: #4b5563 !important;
        }

        h1, h2, h3, h4, h5 {
            color: #047857 !important;
        }
        h2 {
            font-size: 26px !important;
            font-weight: 800 !important;
        }

        p, span, label, div, .stMarkdown, .stText, .stCaption {
            color: #1f2933 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            padding-bottom: 4px !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #4b5563 !important;
            font-weight: 500 !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #047857 !important;
            font-weight: 700 !important;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #10b981 !important;
        }
        div[data-testid="stTabs"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* KPI grid & kartlar */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 20px;
            width: 100%;
            margin-top: 10px;
            margin-bottom: 6px;
        }
        .kpi-card {
            background: #ffffff !important;
            border-radius: 18px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 18px 40px rgba(148,163,184,0.45) !important;
            padding: 0.9rem 1.3rem;
        }
        .kpi-card h3 {
            margin: 0;
            font-size: 0.98rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: #111827 !important;
        }
        .kpi-card .value {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0.25rem 0 0.15rem 0;
            color: #0f172a !important;
        }
        .kpi-card p {
            margin: 0;
            font-size: 0.82rem;
            color: #6b7280 !important;
        }

        /* Butonlar */
        .stButton button {
            background: linear-gradient(135deg,#10b981,#22c55e) !important;
            color: #022c22 !important;
            font-weight: 800 !important;
            border-radius: 999px !important;
            border: none !important;
            box-shadow: 0 10px 28px rgba(16,185,129,0.45) !important;
            padding: 0.55rem 1.5rem !important;
            transition: transform 0.13s ease-out, box-shadow 0.13s ease-out;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 40px rgba(16,185,129,0.6) !important;
        }

        /* Input & select */
        input, textarea {
            background: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #cbd5e1 !important;
            color: #111827 !important;
        }
        input:hover, textarea:hover {
            border-color: #10b981 !important;
        }

        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border-radius: 10px !important;
            border: 1px solid #cbd5e1 !important;
        }
        div[data-baseweb="select"] * {
            color: #111827 !important;
        }
        ul[role="listbox"] {
            background: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Slider */
        .stSlider > div[data-baseweb="slider"] > div > div {
            background: #e5e7eb !important;
        }
        .stSlider [role="slider"] {
            background: #f97316 !important;
            box-shadow: 0 0 0 3px rgba(248,113,113,0.25) !important;
        }

        /* plotly title */
        .js-plotly-plot .plotly .gtitle {
            fill: #111827 !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# =========================================================
# KPI GRID (HTML kullanıyor)
# =========================================================
def create_kpi(title, value, subtext="", icon="📊"):
    st.markdown(
        f"""
        <div class="kpi-card">
            <h3><span class="kpi-icon">{icon}</span>{title}</h3>
            <p class="value">{value}</p>
            <p>{subtext}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DATA & HEALTH SCORE
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    def health_score(row):
        score = 50

        c = row["calories"]
        if c <= 100:
            score += 15
        elif c <= 200:
            score += 8
        elif c >= 350:
            score -= 10

        p = row["protein"]
        if p >= 15:
            score += 18
        elif p >= 8:
            score += 10
        elif p >= 4:
            score += 5

        f = row["fat"]
        if f >= 20:
            score -= 15
        elif f >= 10:
            score -= 7

        vitc = 0 if pd.isna(row.get("vitamin_c", 0)) else row.get("vitamin_c", 0)
        if vitc >= 30:
            score += 10
        elif vitc >= 10:
            score += 6

        iron = 0 if pd.isna(row.get("iron", 0)) else row.get("iron", 0)
        if iron >= 3:
            score += 6
        elif iron >= 1.5:
            score += 3

        return max(0, min(100, score))

    df["health_score"] = df.apply(health_score, axis=1)
    return df


df = load_data()

# =========================================================
# ML – CALORIE PREDICTION (RIDGE + CV)
# =========================================================
@st.cache_resource
def train_calorie_model(data: pd.DataFrame):
    X = data[["protein", "carbs", "fat"]].values
    y = data["calories"].values

    alphas = [0.1, 1.0, 10.0]
    best_alpha = alphas[0]
    best_score = -np.inf
    cv_scores = []

    for a in alphas:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=a)),
            ]
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            pipe.fit(X[train_idx], y[train_idx])
            scores.append(pipe.score(X[val_idx], y[val_idx]))
        mean_r2 = np.mean(scores)
        cv_scores.append((a, mean_r2))
        if mean_r2 > best_score:
            best_score = mean_r2
            best_alpha = a

    best_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=best_alpha)),
        ]
    )
    best_model.fit(X, y)

    return best_model, cv_scores, best_alpha, best_score


cal_model, cv_scores, best_alpha, best_cv = train_calorie_model(df)

# =========================================================
# CLUSTERING – KMEANS + PCA
# =========================================================
def compute_clusters(data: pd.DataFrame, n_clusters: int):
    feats = data[["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(feats)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(feats)

    result = data.copy()
    result["cluster"] = clusters
    result["pc1"] = pcs[:, 0]
    result["pc2"] = pcs[:, 1]

    explained = pca.explained_variance_ratio_.sum()
    return result, explained


# =========================================================
# SIMILARITY – RECOMMENDER
# =========================================================
@st.cache_resource
def build_similarity_matrix(data: pd.DataFrame):
    feats = data[["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]].fillna(0)
    sim = cosine_similarity(feats)
    return sim


sim_matrix = build_similarity_matrix(df)

# =========================================================
# CATEGORY STATE
# =========================================================
categories = sorted(df["category"].unique())
if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = categories.copy()
selected_categories = st.session_state.selected_categories

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    # Language selection first (label sabit)
    lang_choice = st.radio(
        "Dil / Language",
        options=list(LANG_MAP.keys()),
        index=0,
    )
    lang = LANG_MAP[lang_choice]

    st.markdown(f"### 🧭 {t('sidebar_control', lang)}")

    theme_choice = st.radio(
        label=t("sidebar_theme", lang),
        options=[t("theme_dark", lang), t("theme_light", lang)],
        index=0,
    )
    theme = "Dark" if theme_choice == t("theme_dark", lang) else "Light"
    inject_css(theme)

    st.markdown("---")

    st.markdown(f"### {t('sidebar_categories', lang)}")

    selected_count = len(selected_categories)
    if selected_count == 0:
        st.caption(t("no_results", lang))
    else:
        st.caption(f"**{selected_count} {t('sidebar_categories_selected', lang)}**")

    with st.expander(t("sidebar_open_category_list", lang)):
        select_all = st.checkbox(
            t("sidebar_select_all", lang),
            value=(selected_count == len(categories)),
            key="select_all_categories",
        )

        if select_all:
            new_selection = st.multiselect(
                t("sidebar_categories", lang),
                categories,
                default=categories,
                key="multi_cat",
            )
        else:
            new_selection = st.multiselect(
                t("sidebar_categories", lang),
                categories,
                default=selected_categories,
                key="multi_cat",
            )

        if st.button("Apply" if lang == "en" else "Uygula", key="apply_categories"):
            st.session_state.selected_categories = new_selection
            st.experimental_rerun()

    min_cal, max_cal = int(df["calories"].min()), int(df["calories"].max())
    calorie_range = st.slider(
        t("sidebar_calorie_range", lang),
        min_value=min_cal,
        max_value=max_cal,
        value=(min_cal, max_cal),
        step=10,
    )

    focus = st.radio(
        t("sidebar_focus", lang),
        options=[
            t("focus_all", lang),
            t("focus_high_protein", lang),
            t("focus_low_carb", lang),
            t("focus_low_fat", lang),
        ],
    )

    cluster_k = st.slider(
        "K-Means clusters (ML Lab)",
        min_value=2,
        max_value=8,
        value=4,
    )

# =========================================================
# FILTERED DATA
# =========================================================
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df["category"].isin(selected_categories))
    & (filtered_df["calories"].between(calorie_range[0], calorie_range[1]))
]

if focus == t("focus_high_protein", lang):
    filtered_df = filtered_df[filtered_df["protein"] >= filtered_df["protein"].median()]
    filtered_df = filtered_df.sort_values("protein", ascending=False)
elif focus == t("focus_low_carb", lang):
    filtered_df = filtered_df[filtered_df["carbs"] <= filtered_df["carbs"].median()]
    filtered_df = filtered_df.sort_values("carbs", ascending=True)
elif focus == t("focus_low_fat", lang):
    filtered_df = filtered_df[filtered_df["fat"] <= filtered_df["fat"].median()]
    filtered_df = filtered_df.sort_values("fat", ascending=True)

# =========================================================
# HEADER – 🍏 BANNER
# =========================================================
st.markdown(
    f"""
    <div class="app-header-banner">
        <div class="app-header-left">
            <div class="app-logo-chip">
                <div class="app-logo-inner">
                    <span class="app-logo-emoji">🍏</span>
                </div>
            </div>
            <div class="app-header-text">
                <div class="app-title">{t('app_title', lang)}</div>
                <div class="app-subtitle">{t('subtitle', lang)}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        f"📊 {t('overview_tab', lang)}",
        f"🔍 {t('explorer_tab', lang)}",
        f"⚖️ {t('compare_tab', lang)}",
        f"🧪 {t('ml_tab', lang)}",
        f"✨ {t('smartpicks_tab', lang)}",
        f"🤝 {t('recommender_tab', lang)}",
    ]
)

# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    if filtered_df.empty:
        st.warning(t("no_results", lang))
    else:
        total_foods = len(filtered_df)
        total_categories = filtered_df["category"].nunique()
        median_cal = int(filtered_df["calories"].median())
        top_health_row = filtered_df.sort_values("health_score", ascending=False).iloc[0]
        top_health_score = int(top_health_row["health_score"])
        top_health_food = top_health_row["food_name"]

        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
        create_kpi(t("kpi_total_foods", lang), total_foods, t("table_food", lang), "📦")
        create_kpi(t("kpi_total_categories", lang), total_categories, "unique", "🗂️")
        create_kpi(t("kpi_median_calories", lang), median_cal, nl("calories", lang), "🔥")
        create_kpi(t("kpi_top_health", lang), top_health_score, top_health_food, "💚")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown(f"#### {t('section_category_macros', lang)}")
        cat_macro = (
            filtered_df.groupby("category")[["protein", "carbs", "fat"]]
            .mean()
            .reset_index()
        )
        cat_macro_melt = cat_macro.melt(
            id_vars="category",
            value_vars=["protein", "carbs", "fat"],
            var_name="nutrient",
            value_name="value",
        )

        fig_macro = px.bar(
            cat_macro_melt,
            x="category",
            y="value",
            color="nutrient",
            barmode="group",
            labels={"category": "Category", "value": "Avg (g)", "nutrient": "Nutrient"},
        )
        fig_macro.update_layout(legend_title_text="Nutrient")
        st.plotly_chart(fig_macro, use_container_width=True)

        st.markdown("")
        st.markdown(f"#### {t('section_calorie_protein_scatter', lang)}")
        fig_scatter = px.scatter(
            filtered_df,
            x="calories",
            y="protein",
            color="category",
            hover_data=["food_name", "carbs", "fat", "health_score"],
            labels={
                "calories": nl("calories", lang),
                "protein": nl("protein", lang),
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================
# TAB 2 – EXPLORER
# =========================================================
with tab2:
    st.markdown(f"### {t('explorer_title', lang)}")
    st.markdown(t("explorer_description", lang))

    if filtered_df.empty:
        st.warning(t("no_results", lang))
    else:
        nut_choice = st.selectbox(
            t("explorer_select_nutrient", lang),
            options=[
                "calories",
                "protein",
                "carbs",
                "fat",
                "iron",
                "vitamin_c",
                "health_score",
            ],
            format_func=lambda x: nl(x, lang),
        )

        cA, cB = st.columns((2, 3))

        with cA:
            st.markdown(f"#### {t('table_food', lang)}")
            show_cols = [
                "food_name",
                "category",
                "calories",
                "protein",
                "carbs",
                "fat",
                "iron",
                "vitamin_c",
                "health_score",
            ]
            st.dataframe(
                filtered_df[show_cols].reset_index(drop=True),
                use_container_width=True,
                height=500,
            )

        with cB:
            st.markdown(f"#### {nl(nut_choice, lang)}")

            cat_counts = filtered_df["category"].value_counts()
            n_cats = len(cat_counts)
            max_display = min(12, n_cats)

            best_k = None
            add_other = False

            for k in range(1, max_display):
                top = cat_counts.iloc[:k]
                other = cat_counts.iloc[k:].sum()

                if other == 0:
                    best_k = k
                    add_other = False
                    break

                if other <= top.min():
                    best_k = k
                    add_other = True
                    break

            if best_k is None:
                best_k = max_display
                add_other = False

            top_cats = cat_counts.iloc[:best_k].index

            if add_other:
                df_plot = filtered_df.copy()
                df_plot["category_grouped"] = np.where(
                    df_plot["category"].isin(top_cats),
                    df_plot["category"],
                    "Other Categories",
                )
                ordered_categories = list(top_cats) + ["Other Categories"]
            else:
                df_plot = filtered_df[filtered_df["category"].isin(top_cats)].copy()
                df_plot["category_grouped"] = df_plot["category"]
                ordered_categories = list(top_cats)

            df_plot["category_grouped"] = pd.Categorical(
                df_plot["category_grouped"],
                categories=ordered_categories,
                ordered=True,
            )

            fig_hist = px.histogram(
                df_plot,
                x=nut_choice,
                color="category_grouped",
                nbins=25,
                barmode="overlay",
                opacity=0.75,
                labels={
                    nut_choice: nl(nut_choice, lang),
                    "category_grouped": "Category",
                },
            )
            fig_hist.update_layout(
                legend_title_text="Category (Grouped)",
                bargap=0.05,
            )

            st.plotly_chart(fig_hist, use_container_width=True)

# =========================================================
# TAB 3 – COMPARE
# =========================================================
with tab3:
    st.markdown(f"### {t('compare_title', lang)}")
    st.markdown(t("compare_instruction", lang))

    all_foods = df["food_name"].tolist()
    label_foods = "Gıdalar" if lang == "tr" else "Foods"
    selected_foods = st.multiselect(
        label_foods,
        options=all_foods,
        default=all_foods[:3],
        max_selections=4,
    )

    if len(selected_foods) < 2:
        st.info(t("compare_warning", lang))
    else:
        cmp_df = df[df["food_name"].isin(selected_foods)].copy()
        nutrients = [
            "calories",
            "protein",
            "carbs",
            "fat",
            "iron",
            "vitamin_c",
            "health_score",
        ]

        norm_df = cmp_df.copy()
        for n in nutrients:
            max_val = df[n].max()
            if max_val and not pd.isna(max_val):
                norm_df[n] = (norm_df[n] - df[n].min()) / (
                    df[n].max() - df[n].min()
                ) * 100
            else:
                norm_df[n] = 0

        fig_radar = go.Figure()
        for _, row in norm_df.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[row[n] for n in nutrients],
                    theta=[nl(n, lang) for n in nutrients],
                    fill="toself",
                    name=row["food_name"],
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown(f"#### {t('compare_detail', lang)}")
        st.dataframe(
            cmp_df[["food_name"] + nutrients].reset_index(drop=True),
            use_container_width=True,
        )

# =========================================================
# TAB 4 – ML LAB (PREDICTION + CLUSTERING)
# =========================================================
with tab4:
    st.markdown(f"### {t('ml_title', lang)}")

    cL, cR = st.columns(2)

    with cL:
        st.markdown(f"#### {t('ml_pred_title', lang)}")
        st.caption(t("ml_pred_desc", lang))

        st.write(f"Best alpha: **{best_alpha}**, CV mean R²: **{best_cv:.3f}**")

        p_val = st.number_input(nl("protein", lang), 0.0, 100.0, 10.0, step=1.0)
        c_val = st.number_input(nl("carbs", lang), 0.0, 150.0, 10.0, step=1.0)
        f_val = st.number_input(nl("fat", lang), 0.0, 100.0, 5.0, step=1.0)

        if st.button(t("ml_predict_button", lang), key="pred_button"):
            pred = cal_model.predict(np.array([[p_val, c_val, f_val]]))[0]
            if lang == "tr":
                st.success(f"Tahmini Kalori: **{pred:.1f} kcal**")
            else:
                st.success(f"Estimated Calories: **{pred:.1f} kcal**")

    with cR:
        st.markdown(f"#### {t('ml_cluster_title', lang)}")
        st.caption(t("ml_cluster_desc", lang))

        clustered_df, explained = compute_clusters(df, cluster_k)
        st.write(f"PCA variance explained: **{explained*100:.1f}%**")

        title_cluster = "Gıda Besin Haritası (PCA + K-Means)" if lang == "tr" else "Food Nutrition Map (PCA + K-Means)"
        fig_cluster = px.scatter(
            clustered_df,
            x="pc1",
            y="pc2",
            color="cluster",
            hover_name="food_name",
            hover_data=["category", "calories", "protein", "carbs", "fat"],
            title=title_cluster,
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

# =========================================================
# TAB 5 – SMART PICKS
# =========================================================
with tab5:
    st.markdown(f"### {t('smartpicks_title', lang)}")
    st.markdown(t("smartpicks_desc", lang))

    mode = st.radio(
        t("smartpicks_mode", lang),
        options=[
            t("smart_high_protein", lang),
            t("smart_low_calorie", lang),
            t("smart_high_iron", lang),
            t("smart_vitc", lang),
        ],
        horizontal=True,
    )

    smart_df = filtered_df.copy()

    if mode == t("smart_high_protein", lang):
        smart_df["protein_density"] = smart_df["protein"] / smart_df["calories"]
        smart_df = smart_df.dropna(subset=["protein_density"])
        smart_df = smart_df.sort_values("protein_density", ascending=False)
    elif mode == t("smart_low_calorie", lang):
        smart_df = smart_df.sort_values("calories", ascending=True)
    elif mode == t("smart_high_iron", lang):
        smart_df = smart_df[smart_df["iron"] >= 2]
        smart_df = smart_df.sort_values(
            ["iron", "health_score"],
            ascending=[False, False],
        )
    elif mode == t("smart_vitc", lang):
        smart_df["vitc_density"] = smart_df["vitamin_c"] / smart_df["calories"]
        smart_df = smart_df.dropna(subset=["vitc_density"])
        smart_df = smart_df.sort_values("vitc_density", ascending=False)

    if smart_df.empty:
        st.warning(t("no_results", lang))
    else:
        if lang == "tr":
            st.write(f"Toplam **{len(smart_df)}** gıda bulundu.")
        else:
            st.write(f"Found **{len(smart_df)}** items.")
        st.dataframe(
            smart_df[
                [
                    "food_name",
                    "category",
                    "calories",
                    "protein",
                    "carbs",
                    "fat",
                    "iron",
                    "vitamin_c",
                    "health_score",
                ]
            ].reset_index(drop=True),
            use_container_width=True,
        )

# =========================================================
# TAB 6 – RECOMMENDER
# =========================================================
with tab6:
    st.markdown(f"### {t('recom_title', lang)}")
    st.markdown(t("recom_desc", lang))

    food_list = df["food_name"].tolist()
    label_food = "Gıda" if lang == "tr" else "Food"
    base_food = st.selectbox(label_food, options=food_list)

    top_n = st.slider(
        t("recom_howmany", lang),
        min_value=3,
        max_value=15,
        value=8,
    )

    if st.button(t("recom_button", lang), key="recom_button"):
        idx = df[df["food_name"] == base_food].index[0]
        sims = sim_matrix[idx]

        temp = df.copy()
        temp["similarity"] = sims
        temp = temp[temp["food_name"] != base_food]
        result = temp.sort_values("similarity", ascending=False).head(top_n)

        st.dataframe(
            result[
                [
                    "food_name",
                    "category",
                    "calories",
                    "protein",
                    "carbs",
                    "fat",
                    "iron",
                    "vitamin_c",
                    "health_score",
                    "similarity",
                ]
            ].reset_index(drop=True),
            use_container_width=True,
        )

        title_sim = "En Benzer Gıdalar (Kosinüs Benzerliği)" if lang == "tr" else "Most Similar Foods (Cosine Similarity)"
        fig_bar = px.bar(
            result,
            x="food_name",
            y="similarity",
            title=title_sim,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
