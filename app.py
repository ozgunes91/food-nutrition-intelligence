import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Food Nutrition Intelligence",
    layout="wide",
    page_icon="🍏"
)

# -------------------------
# TRANSLATION LAYER
# -------------------------
TEXT = {
    "tr": {
        "app_title": "Food Nutrition Intelligence – Besin Zekâ Paneli",
        "sidebar_language": "Dil / Language",
        "sidebar_theme": "Tema",
        "sidebar_filters": "Filtreler",
        "sidebar_category": "Kategori seç (opsiyonel)",
        "sidebar_calorie_range": "Kalori aralığı (kcal)",
        "sidebar_focus": "Besin odağı",
        "focus_all": "Hepsi",
        "focus_high_protein": "Yüksek Protein",
        "focus_low_carb": "Düşük Karbonhidrat",
        "focus_low_fat": "Düşük Yağ",
        "kpi_total_foods": "Toplam Gıda",
        "kpi_cat_count": "Kategori Sayısı",
        "kpi_med_cal": "Medyan Kalori",
        "kpi_best_health": "En İyi Sağlık Skoru",
        "tab_overview": "Genel Bakış",
        "tab_explorer": "Keşif",
        "tab_compare": "Karşılaştır",
        "tab_smart": "Akıllı Seçimler",
        "tab_ml": "ML Lab",
        "tab_reco": "Tavsiye Sistemi",
        "tab_about": "Hakkında",
        "overview_macro_title": "Kategori Bazlı Ortalama Makro Dağılımı",
        "overview_scatter_title": "Kalori vs Protein",
        "overview_cat_share": "Kategori Dağılımı",
        "explorer_hist_title": "Besin Dağılımı",
        "explorer_nutrient_select": "İncelenecek besin:",
        "compare_select": "Karşılaştırmak için gıda seç (en fazla 4):",
        "compare_radar_title": "Besin Radar Grafiği",
        "smart_title": "Akıllı Seçimler – Bilimsel Besin Analizi",
        "smart_mode": "Mod seç:",
        "smart_mode_hp": "Yüksek Protein & Düşük Yağ",
        "smart_mode_lowcal": "Düşük Kalorili",
        "smart_mode_iron": "Demirden Zengin",
        "smart_mode_vitc": "Vitamin C Yüksek",
        "smart_results": "Toplam sonuç: {n}",
        "smart_top10": "İlk 10 sonuç (bilimsel sıralı)",
        "ml_cluster_title": "Kümeleme – K-Means + PCA",
        "ml_cluster_slider": "Küme sayısı (K-Means)",
        "ml_cal_title": "Kalori Tahmin Modeli (Ridge Regression)",
        "ml_cal_explain": "Protein, karbonhidrat ve yağ değerlerinden kalori tahmini.",
        "ml_cal_cv": "5-katlı CV R² skoru (ortalama): {score:.3f}",
        "ml_cal_input": "Manuel giriş – makrolardan kalori tahmini",
        "ml_protein": "Protein (g)",
        "ml_carbs": "Karbonhidrat (g)",
        "ml_fat": "Yağ (g)",
        "ml_predict": "Kaloriyi Tahmin Et",
        "reco_title": "Tavsiye Sistemi – Benzer Gıdaları Bul",
        "reco_select": "Referans gıda seç:",
        "reco_results": "En benzer 10 gıda",
        "about_title": "Hakkında",
        "about_text": (
            "Bu dashboard, USDA FoodData Central tabanlı 200+ günlük gıdanın "
            "besin bileşimini analiz etmek için geliştirilmiş premium bir besin zekâ aracıdır. "
            "Filtreler, ML modeli, kümeler ve tavsiye sistemi ile gıda mühendisliği, "
            "diyetetik ve veri bilimi perspektifini birleştirir."
        ),
        "no_data": "Seçilen filtrelerle eşleşen kayıt yok. Filtreleri genişletmeyi deneyin."
    },
    "en": {
        "app_title": "Food Nutrition Intelligence – Nutrition Intelligence Panel",
        "sidebar_language": "Language / Dil",
        "sidebar_theme": "Theme",
        "sidebar_filters": "Filters",
        "sidebar_category": "Select category (optional)",
        "sidebar_calorie_range": "Calorie range (kcal)",
        "sidebar_focus": "Nutrient focus",
        "focus_all": "All",
        "focus_high_protein": "High Protein",
        "focus_low_carb": "Low Carb",
        "focus_low_fat": "Low Fat",
        "kpi_total_foods": "Total Foods",
        "kpi_cat_count": "Category Count",
        "kpi_med_cal": "Median Calories",
        "kpi_best_health": "Best Health Score",
        "tab_overview": "Overview",
        "tab_explorer": "Explorer",
        "tab_compare": "Compare",
        "tab_smart": "Smart Picks",
        "tab_ml": "ML Lab",
        "tab_reco": "Recommender",
        "tab_about": "About",
        "overview_macro_title": "Average Macro Distribution by Category",
        "overview_scatter_title": "Calories vs Protein",
        "overview_cat_share": "Category Share",
        "explorer_hist_title": "Nutrient Distribution",
        "explorer_nutrient_select": "Select nutrient to explore:",
        "compare_select": "Select foods to compare (up to 4):",
        "compare_radar_title": "Nutrition Radar Chart",
        "smart_title": "Smart Picks – Scientific Nutrition Analysis",
        "smart_mode": "Choose mode:",
        "smart_mode_hp": "High Protein & Low Fat",
        "smart_mode_lowcal": "Low Calorie",
        "smart_mode_iron": "Iron-Rich",
        "smart_mode_vitc": "High Vitamin C",
        "smart_results": "Total results: {n}",
        "smart_top10": "Top 10 results (scientific ranking)",
        "ml_cluster_title": "Clustering – K-Means + PCA",
        "ml_cluster_slider": "Number of clusters (K-Means)",
        "ml_cal_title": "Calorie Prediction Model (Ridge Regression)",
        "ml_cal_explain": "Predict calories from protein, carbs and fat values.",
        "ml_cal_cv": "5-fold CV R² score (mean): {score:.3f}",
        "ml_cal_input": "Manual input – predict calories from macros",
        "ml_protein": "Protein (g)",
        "ml_carbs": "Carbs (g)",
        "ml_fat": "Fat (g)",
        "ml_predict": "Predict Calories",
        "reco_title": "Recommender – Find Similar Foods",
        "reco_select": "Select a reference food:",
        "reco_results": "Top 10 most similar foods",
        "about_title": "About",
        "about_text": (
            "This dashboard is a premium nutrition intelligence tool built on "
            "USDA FoodData Central data for 200+ everyday foods. It combines "
            "filters, ML models, clustering and recommenders to support "
            "diet planning, health analysis and food-tech innovation."
        ),
        "no_data": "No records match the selected filters. Try relaxing your filters."
    }
}

def t(key: str, lang: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, key)

# -------------------------
# THEME CSS
# -------------------------
def inject_css(theme: str):
    if theme == "Dark":
        bg_css = """
        <style>
        body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left,#020617 0,#020617 35%,#020617 100%) !important;
            color: #e5e7eb !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom,#020617,#020617) !important;
            color:#e5e7eb !important;
        }
        .kpi-card {
            background: rgba(15,23,42,0.96);
            border-radius: 1.5rem;
            border: 1px solid #1e293b;
            box-shadow: 0 18px 40px rgba(15,23,42,0.85);
            padding: 1rem 1.2rem;
        }
        </style>
        """
    else:
        bg_css = """
        <style>
        body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom,#f9fafb,#e5e7eb) !important;
            color: #111827 !important;
        }
        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            color:#111827 !important;
        }
        .kpi-card {
            background: #ffffff;
            border-radius: 1.5rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 30px rgba(15,23,42,0.08);
            padding: 1rem 1.2rem;
        }
        </style>
        """
    st.markdown(bg_css, unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "Food_Nutrition_Dataset.csv")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.strip()

    # basic cleaning
    num_cols = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]
    df[num_cols] = df[num_cols].fillna(0)

    # health score
    df["health_score"] = (
        df["protein"] * 1.5
        + df["vitamin_c"] * 1.2
        + df["iron"] * 1.1
        - df["fat"] * 0.8
        - df["calories"] * 0.05
    )

    # densities (per calorie) – scientific enrichment
    safe_cal = df["calories"].replace(0, np.nan)
    df["protein_density"] = df["protein"] / safe_cal
    df["iron_density"] = df["iron"] / safe_cal
    df["vitc_density"] = df["vitamin_c"] / safe_cal

    # normalized features for radar
    for col in ["calories", "protein", "carbs", "fat", "iron", "vitamin_c", "health_score"]:
        df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)

    return df

df = load_data()

# -------------------------
# SIDEBAR – LANGUAGE & THEME
# -------------------------
with st.sidebar:
    lang = st.radio("Dil / Language", ["tr", "en"], index=0, format_func=lambda x: "Türkçe" if x=="tr" else "English")
    theme = st.radio(t("sidebar_theme", lang), ["Dark", "Light"], index=0)

inject_css(theme)

st.title(t("app_title", lang))

# -------------------------
# SIDEBAR – FILTERS
# -------------------------
st.sidebar.markdown(f"### {t('sidebar_filters', lang)}")

categories = sorted(df["category"].unique())
selected_categories = st.sidebar.multiselect(
    t("sidebar_category", lang),
    options=categories,
    default=[]
)

cal_min = int(df["calories"].min())
cal_max = int(df["calories"].max())
cal_range = st.sidebar.slider(
    t("sidebar_calorie_range", lang),
    min_value=cal_min,
    max_value=cal_max,
    value=(cal_min, cal_max)
)

focus_options_tr = {
    "Hepsi": "all",
    "Yüksek Protein": "high_protein",
    "Düşük Karbonhidrat": "low_carb",
    "Düşük Yağ": "low_fat",
}
focus_options_en = {
    "All": "all",
    "High Protein": "high_protein",
    "Low Carb": "low_carb",
    "Low Fat": "low_fat",
}
if lang == "tr":
    focus_label_map = focus_options_tr
else:
    focus_label_map = focus_options_en

focus_label = st.sidebar.radio(
    t("sidebar_focus", lang),
    list(focus_label_map.keys()),
    index=0
)
focus_mode = focus_label_map[focus_label]

# -------------------------
# APPLY FILTERS
# -------------------------
filtered_df = df.copy()

if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

filtered_df = filtered_df[
    (filtered_df["calories"] >= cal_range[0]) &
    (filtered_df["calories"] <= cal_range[1])
]

if focus_mode == "high_protein":
    filtered_df = filtered_df[filtered_df["protein"] >= filtered_df["protein"].median()]
elif focus_mode == "low_carb":
    filtered_df = filtered_df[filtered_df["carbs"] <= filtered_df["carbs"].median()]
elif focus_mode == "low_fat":
    filtered_df = filtered_df[filtered_df["fat"] <= filtered_df["fat"].median()]

if filtered_df.empty:
    st.warning(t("no_data", lang))
    st.stop()

# -------------------------
# KPI CARDS – FILTER-RESPONSIVE
# -------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(t("kpi_total_foods", lang), len(filtered_df))
    st.markdown('</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(t("kpi_cat_count", lang), filtered_df["category"].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(t("kpi_med_cal", lang), int(filtered_df["calories"].median()))
    st.markdown('</div>', unsafe_allow_html=True)

with k4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    best_idx = filtered_df["health_score"].idxmax()
    best_food = filtered_df.loc[best_idx, "food_name"]
    st.metric(t("kpi_best_health", lang), best_food)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# TABS
# -------------------------
tab_overview, tab_explorer, tab_compare, tab_smart, tab_ml, tab_reco, tab_about = st.tabs([
    t("tab_overview", lang),
    t("tab_explorer", lang),
    t("tab_compare", lang),
    t("tab_smart", lang),
    t("tab_ml", lang),
    t("tab_reco", lang),
    t("tab_about", lang),
])

# -------------------------
# OVERVIEW TAB
# -------------------------
with tab_overview:
    st.subheader(t("overview_macro_title", lang))

    macro = (
        filtered_df
        .groupby("category")[["protein", "carbs", "fat"]]
        .mean()
        .reset_index()
        .sort_values("protein", ascending=False)
    )

    fig_macro = px.bar(
        macro,
        x="category",
        y=["protein", "carbs", "fat"],
        barmode="group",
        template="plotly_white",
    )
    fig_macro.update_layout(legend_orientation="h", legend_yanchor="bottom", legend_y=1.02)
    st.plotly_chart(fig_macro, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("overview_scatter_title", lang))
        fig_sc = px.scatter(
            filtered_df,
            x="calories",
            y="protein",
            color="category",
            hover_name="food_name",
            size="health_score",
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    with c2:
        st.subheader(t("overview_cat_share", lang))
        cat_counts = filtered_df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig_pie = px.pie(cat_counts, names="category", values="count")
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------
# EXPLORER TAB
# -------------------------
with tab_explorer:
    st.subheader(t("tab_explorer", lang))

    nutr_options = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]
    nutr_labels = {
        "calories": "Calories",
        "protein": "Protein (g)",
        "carbs": "Carbs (g)",
        "fat": "Fat (g)",
        "iron": "Iron (mg)",
        "vitamin_c": "Vitamin C (mg)",
    }
    nutr_choice = st.selectbox(
        t("explorer_nutrient_select", lang),
        nutr_options,
        format_func=lambda x: nutr_labels[x]
    )

    e1, e2 = st.columns(2)

    with e1:
        fig_hist = px.histogram(
            filtered_df,
            x=nutr_choice,
            nbins=20,
            color="category",
        )
        fig_hist.update_layout(title=t("explorer_hist_title", lang))
        st.plotly_chart(fig_hist, use_container_width=True)

    with e2:
        fig_box = px.box(
            filtered_df,
            x="category",
            y=nutr_choice,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Data")
    st.dataframe(
        filtered_df[
            ["food_name", "category", "calories", "protein", "carbs", "fat", "iron", "vitamin_c", "health_score"]
        ].sort_values("health_score", ascending=False),
        use_container_width=True,
        height=400
    )

# -------------------------
# COMPARE TAB – RADAR CHART
# -------------------------
with tab_compare:
    st.subheader(t("compare_radar_title", lang))

    compare_foods = st.multiselect(
        t("compare_select", lang),
        options=sorted(df["food_name"].unique()),
        max_selections=4,
    )

    if compare_foods:
        radar_cols = ["calories_norm", "protein_norm", "carbs_norm", "fat_norm", "iron_norm", "vitamin_c_norm", "health_score_norm"]
        radar_labels = ["Calories", "Protein", "Carbs", "Fat", "Iron", "Vitamin C", "Health Score"]
        rad_df = df[df["food_name"].isin(compare_foods)].reset_index(drop=True)

        fig_radar = go.Figure()
        for _, row in rad_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in radar_cols],
                theta=radar_labels,
                fill="toself",
                name=row["food_name"]
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.dataframe(
            rad_df[
                ["food_name", "category", "calories", "protein", "carbs", "fat", "iron", "vitamin_c", "health_score"]
            ],
            use_container_width=True
        )

# -------------------------
# SMART PICKS – SCIENTIFIC
# -------------------------
with tab_smart:
    st.subheader(t("smart_title", lang))

    smart_modes = [
        t("smart_mode_hp", lang),
        t("smart_mode_lowcal", lang),
        t("smart_mode_iron", lang),
        t("smart_mode_vitc", lang),
    ]

    smart_choice = st.radio(t("smart_mode", lang), smart_modes, horizontal=True)

    f = df.copy()
    score_col = None
    title_metric = ""

    if smart_choice == t("smart_mode_hp", lang):
        # High protein & low fat → protein density
        f = f[(f["protein"] > f["protein"].median()) & (f["fat"] < f["fat"].median())]
        f = f.assign(score=f["protein_density"])
        f = f.dropna(subset=["score"]).sort_values("score", ascending=False)
        score_col = "score"
        title_metric = "protein_density"

    elif smart_choice == t("smart_mode_lowcal", lang):
        f = f[f["calories"] < f["calories"].median()]
        f = f.assign(score=-f["calories"]).sort_values("calories")  # lower is better
        score_col = "score"
        title_metric = "calories"

    elif smart_choice == t("smart_mode_iron", lang):
        f = f[f["iron"] > f["iron"].median()]
        f = f.assign(score=f["iron_density"])
        f = f.dropna(subset=["score"]).sort_values("score", ascending=False)
        score_col = "score"
        title_metric = "iron_density"

    elif smart_choice == t("smart_mode_vitc", lang):
        f = f[f["vitamin_c"] > f["vitamin_c"].median()]
        f = f.assign(score=f["vitc_density"])
        f = f.dropna(subset=["score"]).sort_values("score", ascending=False)
        score_col = "score"
        title_metric = "vitc_density"

    st.write(t("smart_results", lang).format(n=len(f)))

    top10 = f.head(10).copy()
    st.markdown(f"**{t('smart_top10', lang)}**")

    display_cols = ["food_name", "category", "calories", "protein", "carbs", "fat", "iron", "vitamin_c", "health_score"]
    if score_col is not None:
        display_cols.append(score_col)

    st.dataframe(top10[display_cols], use_container_width=True)

    if score_col is not None and not top10.empty:
        fig_top = px.bar(
            top10,
            x="food_name",
            y=score_col,
            title=title_metric,
        )
        st.plotly_chart(fig_top, use_container_width=True)

# -------------------------
# ML LAB – CLUSTERING + CALORIE MODEL
# -------------------------
with tab_ml:
    st.subheader(t("ml_cluster_title", lang))

    # clustering on full df (better structure)
    X = df[["protein", "carbs", "fat", "calories"]].values
    n_clusters = st.slider(t("ml_cluster_slider", lang), 2, 8, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    clust_df = df.copy()
    clust_df["cluster"] = clusters
    clust_df["pca1"] = X_pca[:, 0]
    clust_df["pca2"] = X_pca[:, 1]

    fig_cl = px.scatter(
        clust_df,
        x="pca1",
        y="pca2",
        color="cluster",
        hover_name="food_name",
        hover_data=["category", "calories", "protein", "carbs", "fat"],
    )
    st.plotly_chart(fig_cl, use_container_width=True)

    st.markdown("---")
    st.subheader(t("ml_cal_title", lang))
    st.caption(t("ml_cal_explain", lang))

    # calorie model – Ridge regression
    X_cal = df[["protein", "carbs", "fat"]]
    y_cal = df["calories"]

    model = Ridge(alpha=1.0, random_state=42)
    scores = cross_val_score(model, X_cal, y_cal, cv=5, scoring="r2")
    st.write(t("ml_cal_cv", lang).format(score=scores.mean()))

    model.fit(X_cal, y_cal)

    st.markdown(f"**{t('ml_cal_input', lang)}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        p_val = st.number_input(t("ml_protein", lang), min_value=0.0, max_value=200.0, value=20.0, step=1.0)
    with c2:
        c_val = st.number_input(t("ml_carbs", lang), min_value=0.0, max_value=300.0, value=30.0, step=1.0)
    with c3:
        f_val = st.number_input(t("ml_fat", lang), min_value=0.0, max_value=150.0, value=10.0, step=1.0)

    if st.button(t("ml_predict", lang)):
        pred = model.predict([[p_val, c_val, f_val]])[0]
        st.success(f"≈ {pred:.1f} kcal")

# -------------------------
# RECOMMENDER TAB
# -------------------------
with tab_reco:
    st.subheader(t("reco_title", lang))

    food_sel = st.selectbox(
        t("reco_select", lang),
        options=sorted(df["food_name"].unique())
    )

    if food_sel:
        feat_cols = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]
        mat = df[feat_cols].values
        sim = cosine_similarity(mat)
        sim_df = pd.DataFrame(sim, index=df["food_name"], columns=df["food_name"])

        res = (
            sim_df[food_sel]
            .sort_values(ascending=False)
            .iloc[1:11]
            .reset_index()
            .rename(columns={"index": "food_name", food_sel: "similarity"})
        )

        st.markdown(f"**{t('reco_results', lang)}**")
        st.dataframe(res, use_container_width=True)

# -------------------------
# ABOUT TAB
# -------------------------
with tab_about:
    st.subheader(t("about_title", lang))
    st.write(t("about_text", lang))
    st.markdown("---")
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
