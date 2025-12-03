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

DATA_PATH = "/kaggle/input/food-nutrition-dataset-150-everyday-foods/Food_Nutrition_Dataset.csv"

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
# THEME / CSS (Palantir-ish)
# =========================================================
def inject_css(theme: str):
    if theme == "Dark":
        css = """
        <style>

        /* ====== GLOBAL RESET ====== */
        header, .ea3mdgi4 {
            background: transparent !important;
            height: 0px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* ====== BACKGROUND ====== */
        body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left,#0a0f1e 0%,#111827 45%,#0a0f1e 100%) !important;
            color: #e5e7eb !important;
        }

        /* ====== SIDEBAR ====== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom,#0a0f1e,#111827) !important;
            color: #dce7f3 !important; 
        }

        /* ----- Dashboard Header Banner ----- */
        .app-header-banner {
            width: 100%;
            padding: 1.4rem 1.8rem;
            border-radius: 1.1rem;
            background: rgba(15,23,42,0.82);
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 18px 45px rgba(15,23,42,0.85);
            display: flex;
            align-items: center;
            margin-bottom: 1.4rem;
        }

        .app-header-left {
            display: flex;
            align-items: center;
            gap: 1.1rem;
        }

        .app-logo {
            font-size: 2.3rem;
            filter: drop-shadow(0 0 10px rgba(159,255,203,0.55));
        }

        /* ====== DASHBOARD ANA BAŞLIK (EN BÜYÜK) ====== */
        .app-title {
            font-size: 2.35rem !important;   /* artık en büyük başlık */
            font-weight: 900 !important;
            letter-spacing: -0.03em;
            color: #9FFFCB !important;
        }

        .app-subtitle {
            font-size: 1.05rem !important;
            color: #d8dee9 !important;
            opacity: 0.92;
        }

        /* TAB BAŞLIKLARI H2 – KÜÇÜLTÜLDÜ */
        h2 {
            font-size: 26px !important;
            font-weight: 800 !important;
            color: #7EE0B5 !important;
        }

        /* ====== ALT BAŞLIKLAR (#### ) ====== */
        h3, h4 {
            font-size: 20px !important;
            font-weight: 700 !important;
        }

        /* ====== BUTTONS ====== */
        .stButton > button, button[kind="primary"], .stButton button div {
            background-color: #9FFFCB !important;
            color: #0a0f1e !important;   /* siyah */
            font-weight: 800 !important;
            border-radius: 6px !important;
        }

        .stButton > button:hover {
            background-color: #7AEFB2 !important;
            color: #022c22 !important;
        }

        /* ====== PLOTLY GRAPH TITLES ====== */
        .js-plotly-plot .plotly .gtitle {
            font-size: 17px !important;
            font-weight: 600 !important;
        }
        /* ====== GLOBAL METIN RENGI FIX ====== */
        p, label, span, div, .markdown-text-container, .stMarkdown, .stText, .stCaption {
            color: #f4f4f7 !important;
            opacity: 1 !important;
        }
        /* === GENEL GLOBAL RENGİ KESİN DÜZELTEN KOD === */

        /* Ana metin ( gri olan her şey ) */
        html, body, [data-testid="stAppViewContainer"], p, span, label, div, h1, h2, h3, h4, h5 {
            color: #E5E7EB !important;
        }
        
        /* Selectbox iç yazı ve placeholder */
        div[data-baseweb="select"] * {
            color: #E5E7EB !important;
        }
        
        /* Selectbox dropdown içi */
        ul[role="listbox"] li * {
            color: #E5E7EB !important;
        }
        
        /* Selectbox hovered item */
        ul[role="listbox"] li:hover * {
            color: #9FFFCB !important; /* mint hover */
        }
        
        /* Input içi yazı */
        input, textarea {
            color: #E5E7EB !important;
        }
        
        /* Input placeholder (çok gizli bir selector) */
        input::placeholder {
            color: #CBD5E1 !important;
            opacity: 1 !important;
        }
        
        /* Tab başlıkları gri kalıyordu → düzeltme */
        .stTabs [data-baseweb="tab"] p {
            color: #E5E7EB !important;
        }
        
        .stTabs [aria-selected="true"] p {
            color: #9FFFCB !important;
            font-weight: 700 !important;
        }
        
        /* Section başlığı hala koyu gri kalıyordu → FIX */
        h3, h4, h5 {
            color: #9FFFCB !important;
        }
        
        /* Button text kesin siyah olsun */
        .stButton button, button[kind="primary"], div.stButton > button {
            color: #0a0f1e !important;
            font-weight: 800 !important;
        }
        
        /* Light moddaki beyaz tavan FIX */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        /* ================================
           FINAL DARK THEME FIX PACKAGE
           Selectbox, Input, Dropdown, Button
           ================================ */
        
        /* SELECTBOX ana kutu */
        div[data-baseweb="select"] > div {
            background-color: #0f172a !important; /* dark navy */
            border: 1px solid #334155 !important; /* soft slate border */
            color: #e5e7eb !important;
        }
        
        /* SELECTBOX iç yazı */
        div[data-baseweb="select"] * {
            color: #e5e7eb !important;
        }
        
        /* SELECTBOX placeholder */
        div[data-baseweb="select"] div[data-baseweb="select"] span {
            color: #94a3b8 !important; /* gray-400 */
        }
        
        /* SELECTBOX hover border */
        div[data-baseweb="select"] > div:hover {
            border: 1px solid #9fffcba0 !important; /* soft mint glow */
        }
        
        /* DROPDOWN liste arka planı */
        ul[role="listbox"] {
            background-color: #0f172a !important;
            border: 1px solid #334155 !important;
        }
        
        /* DROPDOWN listedeki her item */
        ul[role="listbox"] li {
            background-color: #0f172a !important;
            color: #e5e7eb !important;
        }
        
        /* DROPDOWN hover (çok önemli) */
        ul[role="listbox"] li:hover {
            background-color: #1e293b !important;
            color: #9FFFCB !important;
        }
        
        /* INPUT alanları */
        input, textarea {
            background-color: #0f172a !important;
            border: 1px solid #334155 !important;
            color: #e5e7eb !important;
        }
        
        /* INPUT hover */
        input:hover, textarea:hover {
            border: 1px solid #9fffcba0 !important;
        }
        
        /* INPUT placeholder */
        input::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }
        
        /* BUTTON (primary) */
        .stButton button, div.stButton > button {
            background: linear-gradient(90deg, #9FFFCB, #86efac) !important;
            color: #0f172a !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
        }
        
        /* BUTTON hover */
        .stButton button:hover {
            background: linear-gradient(90deg, #befee0, #a9f5c2) !important;
            color: #020617 !important;
        }
        
        /* BUTTON disabled fix */
        button:disabled {
            opacity: 0.5 !important;
            background-color: #334155 !important;
            color: #94a3b8 !important;
        }

        </style>
        """

    else:
        css = """
        <style>

        header, .ea3mdgi4 {
            background: transparent !important;
            height: 0px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom,#f9fafb,#e5e7eb) !important;
            color: #111827 !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom,#f1f5f9,#e5e7eb) !important;
        }

        .app-header-banner {
            width: 100%;
            padding: 1.4rem 1.8rem;
            border-radius: 1.1rem;
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(209,213,219,0.9);
            box-shadow: 0 18px 45px rgba(148,163,184,0.45);
            display: flex;
            align-items: center;
            margin-bottom: 1.4rem;
        }

        .app-title {
            font-size: 2.35rem !important;
            font-weight: 900 !important;
            color: #047857 !important;  /* koyu yeşil */
        }

        .app-subtitle {
            font-size: 1.05rem !important;
            color: #374151 !important;
        }

        h2 {
            font-size: 26px !important;
            font-weight: 800 !important;
        }

        .stButton > button {
            background-color: #10b981 !important;
            color: white !important;
            font-weight: 800 !important;
        }

        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# =========================================================
# DATA & HEALTH SCORE
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # basit sağlık skoru (domain mantığına dayalı, pure ML değil)
    def health_score(row):
        score = 50

        # Kalori: çok yüksek kalori cezalı, düşük/orta ödüllü
        c = row["calories"]
        if c <= 100:
            score += 15
        elif c <= 200:
            score += 8
        elif c >= 350:
            score -= 10

        # Protein: yüksek protein ödüllü
        p = row["protein"]
        if p >= 15:
            score += 18
        elif p >= 8:
            score += 10
        elif p >= 4:
            score += 5

        # Yağ: çok yağlılar cezalı
        f = row["fat"]
        if f >= 20:
            score -= 15
        elif f >= 10:
            score -= 7

        # Vitamin C: bağışıklık artı puan
        vitc = 0 if pd.isna(row.get("vitamin_c", 0)) else row.get("vitamin_c", 0)
        if vitc >= 30:
            score += 10
        elif vitc >= 10:
            score += 6

        # Demir: anemiye karşı artı puan
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
            scores.append(pipe.score(X[val_idx], y[val_idx]))  # R^2
        mean_r2 = np.mean(scores)
        cv_scores.append((a, mean_r2))
        if mean_r2 > best_score:
            best_score = mean_r2
            best_alpha = a

    # En iyi alpha ile final modeli tüm veride eğit
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
# CATEGORY STATE (sidebar'dan ÖNCE OLMALI!)
# =========================================================
categories = sorted(df["category"].unique())

# İlk yüklemede tüm kategoriler seçili olsun
if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = categories.copy()

selected_categories = st.session_state.selected_categories

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🧭 Control Panel")

    lang_choice = st.radio(
        "Language",
        options=list(LANG_MAP.keys()),
        index=0,
    )
    lang = LANG_MAP[lang_choice]

    theme_choice = st.radio(
        label=t("sidebar_theme", lang),
        options=[t("theme_dark", lang), t("theme_light", lang)],
        index=0,
    )
    theme = "Dark" if theme_choice == t("theme_dark", lang) else "Light"
    inject_css(theme)

    st.markdown("---")

with st.sidebar:
    st.markdown(f"### {t('sidebar_categories', lang)}")

    selected_count = len(selected_categories)
    if selected_count == 0:
        st.caption(t("no_results", lang))
    else:
        st.caption(f"**{selected_count} {t('sidebar_categories_selected', lang)}**")

    # SADECE BU EXPANDER GİZLENİYOR
    with st.expander(t("sidebar_open_category_list", lang)):
        select_all = st.checkbox(
            t("sidebar_select_all", lang),
            value=(selected_count == len(categories)),
            key="select_all_categories"
        )

        if select_all:
            new_selection = st.multiselect(
                t("sidebar_categories", lang),
                categories,
                default=categories,
                key="multi_cat"
            )
        else:
            new_selection = st.multiselect(
                t("sidebar_categories", lang),
                categories,
                default=selected_categories,
                key="multi_cat"
            )

        if st.button("Apply" if lang == "en" else "Uygula", key="apply_categories"):
            st.session_state.selected_categories = new_selection
            st.experimental_rerun()

    # === DİĞER FİLTRELER BURADA VE EXPANDER DIŞINDA ===
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
            <span class="app-logo">🍏</span>
            <div class="app-header-text">
                <div class="app-title">{t('app_title', lang)}</div>
                <div class="app-subtitle">{t('subtitle', lang)}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
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
        c1, c2, c3, c4 = st.columns(4)

        total_foods = len(filtered_df)
        total_categories = filtered_df["category"].nunique()
        median_cal = int(filtered_df["calories"].median())
        top_health_row = filtered_df.sort_values("health_score", ascending=False).iloc[0]
        top_health_score = int(top_health_row["health_score"])
        top_health_food = top_health_row["food_name"]

        with c1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{t('kpi_total_foods', lang)}</div>
                  <div class="kpi-value">{total_foods}</div>
                  <div class="kpi-sub">{t('table_food', lang)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{t('kpi_total_categories', lang)}</div>
                  <div class="kpi-value">{total_categories}</div>
                  <div class="kpi-sub">unique</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{t('kpi_median_calories', lang)}</div>
                  <div class="kpi-value">{median_cal}</div>
                  <div class="kpi-sub">{nl('calories', lang)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{t('kpi_top_health', lang)}</div>
                  <div class="kpi-value">{top_health_score}</div>
                  <div class="kpi-sub">{top_health_food}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
            options=["calories", "protein", "carbs", "fat", "iron", "vitamin_c", "health_score"],
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

    # --- AKILLI 'OTHER' GRUPLAMA ---
    cat_counts = filtered_df["category"].value_counts()
    n_cats = len(cat_counts)

    # En fazla 12 kategori gösterelim
    max_display = min(12, n_cats)

    best_k = None
    add_other = False

    # Sadece ilk max_display içinden bir bölünme arıyoruz
    for k in range(1, max_display):
        top = cat_counts.iloc[:k]          # ana kategoriler
        other = cat_counts.iloc[k:].sum()  # diğerlerinin toplamı

        if other == 0:
            # Zaten hiç kalan yok → OTHER gereksiz
            best_k = k
            add_other = False
            break

        # ŞART: Other toplamı, ana kategoriler içindeki en küçük count'tan küçük/eşit olsun
        if other <= top.min():
            best_k = k
            add_other = True
            break

    if best_k is None:
        # Uygun split bulunamadıysa:
        # - Sadece en büyük max_display kategoriyi göster
        # - "Other Categories" YOK, kimse veriyi çarpıtmıyor
        best_k = max_display
        add_other = False

    top_cats = cat_counts.iloc[:best_k].index

    if add_other:
        # Ana kategoriler + küçüklerin birleştiği bir Other
        df_plot = filtered_df.copy()
        df_plot["category_grouped"] = np.where(
            df_plot["category"].isin(top_cats),
            df_plot["category"],
            "Other Categories",
        )
    else:
        # Sadece en büyük best_k kategori, diğerleri grafikten çıkar
        df_plot = filtered_df[filtered_df["category"].isin(top_cats)].copy()
        df_plot["category_grouped"] = df_plot["category"]

    # --- HISTOGRAM ---
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
    selected_foods = st.multiselect(
        "Foods",
        options=all_foods,
        default=all_foods[:3],
        max_selections=4,
    )

    if len(selected_foods) < 2:
        st.info(t("compare_warning", lang))
    else:
        cmp_df = df[df["food_name"].isin(selected_foods)].copy()
        nutrients = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c", "health_score"]

        # normalize 0-100 for radar
        norm_df = cmp_df.copy()
        for n in nutrients:
            max_val = df[n].max()
            if max_val and not pd.isna(max_val):
                norm_df[n] = (norm_df[n] - df[n].min()) / (df[n].max() - df[n].min()) * 100
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

        st.markdown("#### Detail")
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

    # ---- Prediction
    with cL:
        st.markdown(f"#### {t('ml_pred_title', lang)}")
        st.caption(t("ml_pred_desc", lang))

        st.write(f"Best alpha: **{best_alpha}**, CV mean R²: **{best_cv:.3f}**")

        p = st.number_input(nl("protein", lang), 0.0, 100.0, 10.0, step=1.0)
        c = st.number_input(nl("carbs", lang), 0.0, 150.0, 10.0, step=1.0)
        f = st.number_input(nl("fat", lang), 0.0, 100.0, 5.0, step=1.0)

        if st.button("Predict Calories", key="pred_button"):
            pred = cal_model.predict(np.array([[p, c, f]]))[0]
            st.success(f"Estimated Calories: **{pred:.1f} kcal**")

    # ---- Clustering
    with cR:
        st.markdown(f"#### {t('ml_cluster_title', lang)}")
        st.caption(t("ml_cluster_desc", lang))

        clustered_df, explained = compute_clusters(df, cluster_k)
        st.write(f"PCA variance explained: **{explained*100:.1f}%**")

        fig_cluster = px.scatter(
            clustered_df,
            x="pc1",
            y="pc2",
            color="cluster",
            hover_name="food_name",
            hover_data=["category", "calories", "protein", "carbs", "fat"],
            title="Food Nutrition Map (PCA + K-Means)",
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
        smart_df = smart_df[smart_df["iron"] >= 2]   # Bilimsel eşik
        smart_df = smart_df.sort_values(
            ["iron", "health_score"],
            ascending=[False, False]
        )
    elif mode == t("smart_vitc", lang):
        smart_df["vitc_density"] = smart_df["vitamin_c"] / smart_df["calories"]
        smart_df = smart_df.dropna(subset=["vitc_density"])
        smart_df = smart_df.sort_values("vitc_density", ascending=False)

    if smart_df.empty:
        st.warning(t("no_results", lang))
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
    base_food = st.selectbox("Food", options=food_list)

    top_n = st.slider("How many similar foods?", min_value=3, max_value=15, value=8)

    if st.button("Find Similar Foods", key="recom_button"):
        idx = df[df["food_name"] == base_food].index[0]
        sims = sim_matrix[idx]

        temp = df.copy()
        temp["similarity"] = sims
        # kendisini hariç tut
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

        fig_bar = px.bar(
            result,
            x="food_name",
            y="similarity",
            title="Most Similar Foods (Cosine Similarity)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
