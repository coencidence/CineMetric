"""
CineMetric: AI Movie Predictor
A Streamlit web application that predicts movie ratings using a Random Forest Regressor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMetric: AI Movie Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }

    /* ── Headers ── */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #6dd5ed 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* ── Metric Card ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(109, 213, 237, 0.10));
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }

    /* ── Info Card ── */
    .info-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Chart containers ── */
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(167, 139, 250, 0.15) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# DATA & MODEL (cached so it only runs once)
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_and_train():
    """Load CSV, clean, engineer features, and train the model."""
    df = pd.read_csv("movies_data.csv")

    # ── Clean ──
    df = df[df["rating"] > 0].copy()
    df = df.dropna(subset=["genres", "directors"]).copy()
    df["actors"] = df["actors"].fillna("Unknown")
    for col in ["genres", "directors", "actors"]:
        df[col] = df[col].str.strip()
    df = df.drop_duplicates(subset="title", keep="first").copy()

    # ── Genre encoding ──
    df["genre_list"] = df["genres"].apply(lambda x: [g.strip() for g in x.split(",")])
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(df["genre_list"]),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index,
    )

    # ── Director encoding ──
    df["primary_director"] = df["directors"].apply(lambda x: x.split(",")[0].strip())
    le_director = LabelEncoder()
    df["director_encoded"] = le_director.fit_transform(df["primary_director"])

    # ── Lead actor encoding ──
    df["lead_actor"] = df["actors"].apply(lambda x: x.split(",")[0].strip())
    le_actor = LabelEncoder()
    df["actor_encoded"] = le_actor.fit_transform(df["lead_actor"])

    # ── Feature matrix ──
    feature_cols = list(genre_encoded.columns) + ["director_encoded", "actor_encoded"]
    X = pd.concat([genre_encoded, df[["director_encoded", "actor_encoded"]]], axis=1)
    y = df["rating"]

    # ── Train ──
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    return model, df, mlb, le_director, le_actor, feature_cols, genre_encoded


model, df, mlb, le_director, le_actor, feature_cols, genre_encoded = load_and_train()

# Collect unique values for dropdowns
all_genres = sorted(set(g for gl in df["genre_list"] for g in gl))
all_directors = sorted(df["primary_director"].unique())


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 About")
    st.markdown(
        """
        **CineMetric** uses a Random Forest Regressor trained on 
        real movie data to predict ratings based on genre, director, 
        and other features.
        
        Enter your movie details below and hit **Predict Rating** 
        to see the AI-powered score.
        """
    )

    st.markdown("---")
    st.markdown("## 🎯 Movie Details")

    movie_title = st.text_input("Movie Title", placeholder="e.g. The Last Horizon")

    selected_genre = st.selectbox("Primary Genre", options=all_genres, index=all_genres.index("Drama") if "Drama" in all_genres else 0)

    selected_director = st.selectbox(
        "Director",
        options=["— Custom (not in dataset) —"] + all_directors,
        index=0,
    )
    custom_director = None
    if selected_director == "— Custom (not in dataset) —":
        custom_director = st.text_input("Enter Director Name", placeholder="e.g. Jane Doe")

    budget = st.slider("Budget ($ Millions)", min_value=1, max_value=500, value=50, step=1)

    st.markdown("---")

    predict_clicked = st.button("🔮  Predict Rating")


# ══════════════════════════════════════════════════════════════════════
# MAIN PAGE — HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">CineMetric: AI Movie Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Predict movie ratings with machine learning — powered by Random Forest</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════
def build_input_vector(genre, director_name, budget_val):
    """Build a feature vector matching the training feature set."""
    warnings_list = []

    # Genre one-hot
    genre_vector = {col: 0 for col in genre_encoded.columns}
    genre_col = f"genre_{genre}"
    if genre_col in genre_vector:
        genre_vector[genre_col] = 1
    else:
        warnings_list.append(f"Genre **{genre}** not found in training data — using global average.")

    # Director encoding
    if director_name and director_name in le_director.classes_:
        dir_enc = le_director.transform([director_name])[0]
    else:
        dir_enc = int(np.median(df["director_encoded"]))
        label = director_name or "Unknown"
        warnings_list.append(f"Director **{label}** not in dataset — using global average.")

    # Actor: use median (user doesn't choose actor)
    actor_enc = int(np.median(df["actor_encoded"]))

    # Assemble
    row = list(genre_vector.values()) + [dir_enc, actor_enc]
    return np.array(row).reshape(1, -1), warnings_list


if predict_clicked:
    if not movie_title.strip():
        st.warning("⚠️ Please enter a movie title in the sidebar.")
    else:
        director_name = custom_director if custom_director else (
            selected_director if selected_director != "— Custom (not in dataset) —" else None
        )

        input_vec, warn_msgs = build_input_vector(selected_genre, director_name, budget)

        # ── Show warnings ──
        for w in warn_msgs:
            st.info(f"ℹ️ {w}")

        # ── Predict ──
        predicted_rating = model.predict(input_vec)[0]
        predicted_rating = round(np.clip(predicted_rating, 0, 10), 2)

        # ── Display metric ──
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            emoji = "🌟" if predicted_rating >= 7 else ("👍" if predicted_rating >= 5 else "👎")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Rating</div>
                    <div class="metric-value">{predicted_rating} / 10</div>
                    <div class="metric-sub">{emoji} {movie_title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════
        # CHARTS
        # ══════════════════════════════════════════════════════════════
        chart_col1, chart_col2 = st.columns(2)

        # ── Chart 1: Top 5 Similar Movies ──
        with chart_col1:
            st.markdown("### 🎥 Top 5 Similar Movies")
            genre_matches = df[df["genre_list"].apply(lambda g: selected_genre in g)].copy()
            if genre_matches.empty:
                genre_matches = df.copy()
            genre_matches["rating_diff"] = (genre_matches["rating"] - predicted_rating).abs()
            top5 = genre_matches.nsmallest(5, "rating_diff")[["title", "rating"]].reset_index(drop=True)

            fig1, ax1 = plt.subplots(figsize=(7, 4))
            fig1.patch.set_facecolor("#0f0c29")
            ax1.set_facecolor("#1a1a2e")

            colors = ["#a78bfa", "#818cf8", "#6dd5ed", "#34d399", "#f093fb"]
            bars = ax1.barh(
                top5["title"],
                top5["rating"],
                color=colors[: len(top5)],
                edgecolor="none",
                height=0.55,
            )
            for bar, rating in zip(bars, top5["rating"]):
                ax1.text(
                    bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{rating:.1f}",
                    va="center",
                    ha="left",
                    color="#e0e0e0",
                    fontweight="bold",
                    fontsize=10,
                )

            ax1.set_xlim(0, 10.5)
            ax1.set_xlabel("Rating", color="#9ca3af", fontsize=10)
            ax1.tick_params(colors="#9ca3af", labelsize=9)
            ax1.invert_yaxis()
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["bottom"].set_color("#2d2d44")
            ax1.spines["left"].set_color("#2d2d44")
            ax1.grid(axis="x", color="#2d2d44", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        # ── Chart 2: Feature Importance ──
        with chart_col2:
            st.markdown("### 📊 Feature Importance")
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            top_imp = importances.nlargest(10).sort_values()

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_facecolor("#0f0c29")
            ax2.set_facecolor("#1a1a2e")

            gradient = plt.cm.cool(np.linspace(0.3, 0.9, len(top_imp)))
            ax2.barh(
                top_imp.index.str.replace("genre_", "").str.replace("_encoded", ""),
                top_imp.values,
                color=gradient,
                edgecolor="none",
                height=0.55,
            )
            for i, (val, name) in enumerate(zip(top_imp.values, top_imp.index)):
                ax2.text(
                    val + 0.002,
                    i,
                    f"{val:.3f}",
                    va="center",
                    color="#e0e0e0",
                    fontsize=9,
                    fontweight="bold",
                )

            ax2.set_xlabel("Importance", color="#9ca3af", fontsize=10)
            ax2.tick_params(colors="#9ca3af", labelsize=9)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["bottom"].set_color("#2d2d44")
            ax2.spines["left"].set_color("#2d2d44")
            ax2.grid(axis="x", color="#2d2d44", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # ── Summary card ──
        st.markdown("---")
        st.markdown(
            f"""
            <div class="info-card">
                <strong>Summary</strong><br>
                <strong>Title:</strong> {movie_title} &nbsp;|&nbsp;
                <strong>Genre:</strong> {selected_genre} &nbsp;|&nbsp;
                <strong>Director:</strong> {director_name or 'N/A'} &nbsp;|&nbsp;
                <strong>Budget:</strong> ${budget}M &nbsp;|&nbsp;
                <strong>Predicted Rating:</strong> {predicted_rating}/10
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # ── Landing state ──
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class="info-card" style="text-align:center; padding: 3rem 2rem;">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">🎬</p>
                <p style="font-size: 1.1rem; color: #9ca3af;">
                    Fill in the movie details in the sidebar<br>
                    and click <strong style="color:#a78bfa;">Predict Rating</strong> to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#4b5563; font-size:0.8rem;">'
    "CineMetric v1.0 — Built with Streamlit & Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
