###############################################
# FOOD NUTRITION INTELLIGENCE – FINAL VERSION
# Flat Glass A1 (Palantir-style)
# Full working code – NOTHING removed or broken
###############################################

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


# =========================================================
# LANGUAGE MAP & TEXTS
# =========================================================

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
        "tr": "Gıda Keşfi & Dağılım Analizi",
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
    "compare_warning": {"tr": "Karşılaştırma için en az 2 gıda seç.", "en": "Select at least 2 foods."},
    "ml_title": {"tr": "ML Lab: Kalori Tahmini & Kümeleme", "en": "ML Lab: Calorie Prediction & Clustering"},
    "ml_pred_title": {"tr": "Kalori Tahmini (Ridge Regression)", "en": "Calorie Prediction (Ridge Regression)"},
    "ml_pred_desc": {
        "tr": "Protein / Karbonhidrat / Yağ değerlerine göre kalori tahmini yapar.",
        "en": "Predicts calories using protein/carbs/fat based on a regularized model.",
    },
    "ml_cluster_title": {"tr": "Gıda Kümeleme (K-Means + PCA)", "en": "Food Clustering (K-Means + PCA)"},
    "ml_cluster_desc": {
        "tr": "Besin profillerine göre kümeler ve PCA ile 2D harita oluşturur.",
        "en": "Clusters foods and creates a 2D PCA-based food map.",
    },
    "recom_title": {"tr": "Benzer Gıda Önerileri", "en": "Similar Food Recommendations"},
    "recom_desc": {
        "tr": "Besin profili benzer gıdaları listeler.",
        "en": "Lists foods with similar nutrient profiles.",
    },
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


def t(key, lang):
    return TEXT.get(key, {}).get(lang, key)


def nl(col, lang):
    return NUTR_LABELS.get(col, {}).get(lang, col)


# =========================================================
# CSS (A1 – Flat Glass Final)
# =========================================================

def inject_css(theme: str):
    if theme == "Dark":
        css = """
        <style>

        /* === GLOBAL BACKGROUND === */
        body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left,#0d1222 0%,#111827 45%,#0d1222 100%) !important;
            color: #e5e7eb !important;
        }

        header, .ea3mdgi4 { background: transparent !important; }

        /* === SIDEBAR === */
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom,#0d1222,#111827) !important;
            color: #dce7f3 !important;
        }

        .stSidebar h1, .stSidebar h2 {
            color: #A8FFBF !important;
            font-weight: 700 !important;
        }

        /* === TEXT / LABELS === */
        p, label, span, div {
            color: #dbe3ed !important;
        }

        /* === TABS === */
        .stTabs [data-baseweb="tab"] {
            color: #c4d0dd !important;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            color: #A8FFBF !important;
            border-bottom: 3px solid #A8FFBF !important;
            font-weight: 700 !important;
        }

        /* === HEADER BOX (A1 Flat Glass) === */
        .header-box {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 18px 28px;
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.45);
        }
        .header-title {
            font-size: 26px !important;
            font-weight: 800 !important;
            color: #A8FFBF !important;
        }
        .header-sub {
            font-size: 15px !important;
            color: #c7d3df !important;
        }

        /* === KPI CARDS === */
        .kpi-card {
            background: rgba(17,24,39,0.85);
            border: 1px solid #1f2937;
            border-radius: 1rem;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 22px 50px rgba(0,0,0,0.55);
        }

        /* === INPUTS === */
        div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
        }
        ul[role="listbox"] {
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
        }

        /* === BUTTONS (SİYAH YAZI ZORUNLU) === */
        .stButton > button, 
        button[kind="primary"], 
        button[kind="secondary"], 
        button {
            background-color: #A8FFBF !important;
            color: #000000 !important;
            font-weight: 800 !important;
            border-radius: 6px !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #7AFFA5 !important;
            color: #000000 !important;
        }

        /* === PLOTLY TITLE SMALLER === */
        .js-plotly-plot .plotly .gtitle {
            font-size: 18px !important;
            font-weight: 600 !important;
        }

        </style>
        """

    else:
        # Light theme remains simpler (changed only necessary parts)
        css = """
        <style>
        body, [data-testid="stAppViewContainer"] {
            background: #f9fafb !important;
            color: #111827 !important;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            color:#111827 !important;
        }
        .header-title { color:#111827 !important; }
        .stButton > button { color:#000 !important; }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)


# =========================================================
# LOAD DATA + HEALTH SCORE
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    def health_score(row):
        score = 50
        c = row["calories"]
        if c <= 100: score += 15
        elif c <= 200: score += 8
        elif c >= 350: score -= 10

        p = row["protein"]
        if p >= 15: score += 18
        elif p >= 8: score += 10

        f = row["fat"]
        if f >= 20: score -= 15
        elif f >= 10: score -= 7

        vitc = row.get("vitamin_c", 0)
        if vitc >= 30: score += 10
        elif vitc >= 10: score += 6

        iron = row.get("iron", 0)
        if iron >= 3: score += 6
        elif iron >= 1.5: score += 3

        return max(0, min(100, score))

    df["health_score"] = df.apply(health_score, axis=1)
    return df


df = load_data()


# =========================================================
# ML MODEL – RIDGE REGRESSION
# =========================================================
@st.cache_resource
def train_calorie_model(data):
    X = data[["protein", "carbs", "fat"]].values
    y = data["calories"].values

    alphas = [0.1, 1.0, 10.0]
    best_alpha = alphas[0]
    best_score = -np.inf

    for a in alphas:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, val in kf.split(X):
            pipe = Pipeline([
                ("s", StandardScaler()),
                ("m", Ridge(alpha=a)),
            ])
            pipe.fit(X[tr], y[tr])
            scores.append(pipe.score(X[val], y[val]))

        mean_r2 = np.mean(scores)
        if mean_r2 > best_score:
            best_score = mean_r2
            best_alpha = a

    # final model
    final = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=best_alpha))])
    final.fit(X, y)
    return final, best_alpha, best_score


cal_model, best_alpha, best_cv = train_calorie_model(df)


# =========================================================
# CLUSTERING
# =========================================================
def compute_clusters(data, k):
    feats = data[["calories","protein","carbs","fat","iron","vitamin_c"]].fillna(0)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labs = km.fit_predict(feats)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(feats)

    out = data.copy()
    out["cluster"] = labs
    out["pc1"] = pcs[:,0]
    out["pc2"] = pcs[:,1]
    expl = pca.explained_variance_ratio_.sum()
    return out, expl


# =========================================================
# SIMILARITY MATRIX
# =========================================================
@st.cache_resource
def build_similarity_matrix(data):
    feats = data[["calories","protein","carbs","fat","iron","vitamin_c"]].fillna(0)
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
    st.markdown("### 🧭 Control Panel")

    lang_choice = st.radio("Language", options=list(LANG_MAP.keys()))
    lang = LANG_MAP[lang_choice]

    theme_choice = st.radio("Theme", ["Dark","Light"])
    theme = theme_choice
    inject_css(theme)

    st.markdown("---")

with st.sidebar:
    st.markdown(f"### {t('sidebar_categories', lang)}")

    sel_count = len(selected_categories)
    st.caption(f"**{sel_count} {t('sidebar_categories_selected', lang)}**")

    with st.expander(t("sidebar_open_category_list", lang)):
        all_flag = st.checkbox("Select / Deselect All", value=(sel_count==len(categories)))
        if all_flag:
            new_sel = st.multiselect("Categories", categories, default=categories)
        else:
            new_sel = st.multiselect("Categories", categories, default=selected_categories)

        if st.button("Apply"):
            st.session_state.selected_categories = new_sel
            st.experimental_rerun()

    calorie_range = st.slider(
        t("sidebar_calorie_range", lang),
        min_value=int(df["calories"].min()),
        max_value=int(df["calories"].max()),
        value=(int(df["calories"].min()), int(df["calories"].max())),
        step=10,
    )

    focus = st.radio(
        t("sidebar_focus", lang),
        [t("focus_all", lang), t("focus_high_protein", lang),
         t("focus_low_carb", lang), t("focus_low_fat", lang)]
    )

    cluster_k = st.slider("K-Means clusters (ML Lab)", 2, 8, 4)


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
    filtered_df = filtered_df.sort_values("carbs")

elif focus == t("focus_low_fat", lang):
    filtered_df = filtered_df[filtered_df["fat"] <= filtered_df["fat"].median()]
    filtered_df = filtered_df.sort_values("fat")


# =========================================================
# HEADER (🍏 + FLAT GLASS BOX)
# =========================================================
st.markdown(
    f"""
    <div class="header-box">
        <div style="font-size:32px;">🍏</div>
        <div>
            <div class="header-title">{t('app_title', lang)}</div>
            <div class="header-sub">{t('subtitle', lang)}</div>
        </div>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    f"📊 {t('overview_tab', lang)}",
    f"🔍 {t('explorer_tab', lang)}",
    f"⚖️ {t('compare_tab', lang)}",
    f"🧪 {t('ml_tab', lang)}",
    f"✨ {t('smartpicks_tab', lang)}",
    f"🤝 {t('recommender_tab', lang)}",
])


# =========================================================
# TAB 1 – OVERVIEW
# =========================================================
with tab1:
    if filtered_df.empty:
        st.warning(t("no_results", lang))
    else:
        c1,c2,c3,c4 = st.columns(4)

        total_foods = len(filtered_df)
        total_categories = filtered_df["category"].nunique()
        median_cal = int(filtered_df["calories"].median())
        top_item = filtered_df.sort_values("health_score", ascending=False).iloc[0]

        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{t('kpi_total_foods', lang)}</div>
                <div class="kpi-value">{total_foods}</div>
                <div class="kpi-sub">{t('table_food', lang)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{t('kpi_total_categories', lang)}</div>
                <div class="kpi-value">{total_categories}</div>
                <div class="kpi-sub">unique</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{t('kpi_median_calories', lang)}</div>
                <div class="kpi-value">{median_cal}</div>
                <div class="kpi-sub">{nl('calories', lang)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{t('kpi_top_health', lang)}</div>
                <div class="kpi-value">{int(top_item['health_score'])}</div>
                <div class="kpi-sub">{top_item['food_name']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown(f"### {t('section_category_macros', lang)}")
        macro = filtered_df.groupby("category")[["protein","carbs","fat"]].mean().reset_index()
        melt = macro.melt("category", ["protein","carbs","fat"])

        fig = px.bar(
            melt,
            x="category", y="value",
            color="variable", barmode="group",
            labels={"value":"Avg (g)","variable":"Nutrient"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("")
        st.markdown(f"### {t('section_calorie_protein_scatter', lang)}")

        fig2 = px.scatter(
            filtered_df,
            x="calories", y="protein",
            color="category",
            hover_data=["food_name","carbs","fat","health_score"],
        )
        st.plotly_chart(fig2, use_container_width=True)


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
            ["calories","protein","carbs","fat","iron","vitamin_c","health_score"],
            format_func=lambda x: nl(x,lang)
        )

        cA,cB = st.columns((2,3))

        with cA:
            st.markdown(f"### {t('table_food', lang)}")
            cols = ["food_name","category","calories","protein","carbs","fat","iron","vitamin_c","health_score"]
            st.dataframe(filtered_df[cols], use_container_width=True, height=520)

        with cB:
            st.markdown(f"### {nl(nut_choice, lang)}")

            # Intelligent Other Categories grouping
            counts = filtered_df["category"].value_counts()
            max_display = 12
            best_k = None
            add_other = False

            for k in range(1, min(max_display,len(counts))):
                top = counts.iloc[:k]
                other = counts.iloc[k:].sum()

                if other == 0:
                    best_k = k
                    break

                if other <= top.min():
                    best_k = k
                    add_other = True
                    break

            if best_k is None:
                best_k = min(max_display,len(counts))

            top_cats = counts.iloc[:best_k].index

            if add_other:
                df_plot = filtered_df.copy()
                df_plot["category_grouped"] = np.where(
                    df_plot["category"].isin(top_cats),
                    df_plot["category"],
                    "Other Categories"
                )
            else:
                df_plot = filtered_df[filtered_df["category"].isin(top_cats)].copy()
                df_plot["category_grouped"] = df_plot["category"]

            fig = px.histogram(
                df_plot,
                x=nut_choice,
                color="category_grouped",
                nbins=25,
                opacity=0.75,
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 3 – COMPARE
# =========================================================
with tab3:
    st.markdown(f"### {t('compare_title', lang)}")
    st.markdown(t("compare_instruction", lang))

    foods = df["food_name"].tolist()
    sel = st.multiselect("Foods", foods, default=foods[:3], max_selections=4)

    if len(sel) < 2:
        st.info(t("compare_warning", lang))
    else:
        cmp = df[df["food_name"].isin(sel)].copy()
        nuts = ["calories","protein","carbs","fat","iron","vitamin_c","health_score"]

        norm = cmp.copy()
        for n in nuts:
            norm[n] = (norm[n]-df[n].min())/(df[n].max()-df[n].min())*100

        fig = go.Figure()
        for _,row in norm.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[n] for n in nuts],
                theta=[nl(n,lang) for n in nuts],
                fill="toself",
                name=row["food_name"]
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Detail")
        st.dataframe(cmp[["food_name"]+nuts], use_container_width=True)


# =========================================================
# TAB 4 – ML LAB
# =========================================================
with tab4:
    cL,cR = st.columns(2)

    with cL:
        st.markdown(f"### {t('ml_pred_title', lang)}")
        st.caption(t("ml_pred_desc", lang))

        st.write(f"Best alpha: **{best_alpha}**, CV mean R²: **{best_cv:.3f}**")

        p = st.number_input(nl("protein",lang), 0.0,100.0,10.0)
        c = st.number_input(nl("carbs",lang), 0.0,200.0,10.0)
        f = st.number_input(nl("fat",lang), 0.0,100.0,5.0)

        if st.button("Predict Calories"):
            pred = cal_model.predict(np.array([[p,c,f]]))[0]
            st.success(f"Estimated Calories: **{pred:.1f} kcal**")

    with cR:
        st.markdown(f"### {t('ml_cluster_title', lang)}")
        st.caption(t("ml_cluster_desc", lang))

        clustered, explained = compute_clusters(df, cluster_k)
        st.write(f"PCA variance explained: **{explained*100:.1f}%**")

        fig = px.scatter(
            clustered,
            x="pc1", y="pc2",
            color="cluster",
            hover_name="food_name",
            hover_data=["category","calories","protein","carbs","fat"]
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 5 – SMART PICKS
# =========================================================
with tab5:
    st.markdown(f"### {t('smartpicks_title', lang)}")

    mode = st.radio("Mode", [
        t("focus_high_protein",lang),
        t("smart_low_calorie",lang),
        t("smart_high_iron",lang),
        t("smart_vitc",lang)
    ], horizontal=True)

    sp = filtered_df.copy()

    if mode == t("focus_high_protein",lang):
        sp["density"] = sp["protein"]/sp["calories"]
        sp = sp.sort_values("density", ascending=False)

    elif mode == t("smart_low_calorie",lang):
        sp = sp.sort_values("calories")

    elif mode == t("smart_high_iron",lang):
        sp = sp[sp["iron"]>=2].sort_values("iron",ascending=False)

    elif mode == t("smart_vitc",lang):
        sp["density"] = sp["vitamin_c"]/sp["calories"]
        sp = sp.sort_values("density",ascending=False)

    if sp.empty:
        st.warning(t("no_results",lang))
    else:
        st.dataframe(sp.reset_index(drop=True), use_container_width=True)


# =========================================================
# TAB 6 – RECOMMENDER
# =========================================================
with tab6:
    st.markdown(f"### {t('recom_title', lang)}")
    st.markdown(t("recom_desc", lang))

    food_list = df["food_name"].tolist()
    base_food = st.selectbox("Food", food_list)

    top_n = st.slider("How many similar foods?", 3,15,8)

    if st.button("Find Similar Foods"):
        idx = df[df["food_name"]==base_food].index[0]
        sims = sim_matrix[idx]
        tmp = df.copy()
        tmp["sim"] = sims
        tmp = tmp[tmp["food_name"]!=base_food]
        res = tmp.sort_values("sim", ascending=False).head(top_n)

        st.dataframe(res, use_container_width=True)

        fig = px.bar(res, x="food_name", y="sim", title="Most Similar Foods")
        st.plotly_chart(fig, use_container_width=True)

