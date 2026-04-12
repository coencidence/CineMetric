"""
CineMetric: AI Movie Predictor
A Streamlit web application that predicts movie ratings using a Random Forest Regressor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import re
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMetric: AI Movie Predictor",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&family=Nunito:wght@400;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(150deg, #fff0f6 0%, #fce4ec 35%, #f3e5f5 70%, #e8eaf6 100%);
        color: #4a3f5c;
        font-family: 'Quicksand', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fce4ec 0%, #f8bbd0 100%);
        border-right: 2px solid rgba(236, 64, 122, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c2185b;
        font-family: 'Nunito', sans-serif;
    }
    section[data-testid="stSidebar"] label {
        color: #880e4f !important;
        font-weight: 600 !important;
    }

    /* ── Headers ── */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        font-family: 'Nunito', sans-serif;
        background: linear-gradient(135deg, #f06292 0%, #ce93d8 50%, #90caf9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
        text-shadow: none;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #ad7da0;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* ── Metric Card ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,182,193,0.5), rgba(216,191,216,0.4));
        border: 2px solid rgba(236, 64, 122, 0.2);
        border-radius: 24px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 32px rgba(236, 64, 122, 0.15), 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #c2185b;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 3.8rem;
        font-weight: 800;
        font-family: 'Nunito', sans-serif;
        background: linear-gradient(135deg, #e91e8c, #9c27b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #9c6d8f;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* ── Info / Summary Card ── */
    .info-card {
        background: rgba(255, 255, 255, 0.55);
        border: 1.5px solid rgba(236, 64, 122, 0.18);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(236,64,122,0.07);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #f06292, #ce93d8) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.4px;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(240, 98, 146, 0.35) !important;
        font-family: 'Nunito', sans-serif !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #e91e8c, #ba68c8) !important;
        box-shadow: 0 6px 22px rgba(240, 98, 146, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Input widgets ── */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1.5px solid rgba(236,64,122,0.3) !important;
        background: rgba(255,255,255,0.8) !important;
        color: #4a3f5c !important;
    }
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid rgba(236,64,122,0.3) !important;
        background: rgba(255,255,255,0.8) !important;
    }

    /* ── Chart containers ── */
    .chart-container {
        background: rgba(255, 255, 255, 0.4);
        border: 1.5px solid rgba(236, 64, 122, 0.12);
        border-radius: 18px;
        padding: 1.5rem;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(236, 64, 122, 0.15) !important;
    }

    /* ── Stars badge ── */
    .star-badge {
        display: inline-block;
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPER – synopsis sentiment score
# ══════════════════════════════════════════════════════════════════════
POSITIVE_WORDS = {
    "love", "hope", "dream", "joy", "triumph", "wonder", "beautiful",
    "inspire", "hero", "courage", "victory", "friendship", "magic",
    "adventure", "comedy", "laugh", "fun", "heartwarming", "uplifting",
    "redemption", "discovery"
}
NEGATIVE_WORDS = {
    "death", "murder", "war", "horror", "terror", "dark", "tragedy",
    "betrayal", "evil", "violence", "grief", "destruction", "fear",
    "serial", "killer", "apocalypse", "zombie", "revenge", "conspiracy"
}

def synopsis_score(text: str) -> float:
    """Return a rough sentiment score in [-1, 1] based on keyword presence."""
    if not text or text.strip() == "":
        return 0.0
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


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
    df["synopsis"] = df["synopsis"].fillna("") if "synopsis" in df.columns else ""
    for col in ["genres", "directors", "actors"]:
        df[col] = df[col].str.strip()
    df = df.drop_duplicates(subset="title", keep="first").copy()

    # ── Genre encoding ──
    df["genre_list"] = df["genres"].apply(lambda x: [g.strip() for g in x.split(",")])
    mlb_genre = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb_genre.fit_transform(df["genre_list"]),
        columns=[f"genre_{g}" for g in mlb_genre.classes_],
        index=df.index,
    )

    # ── Actor multi-hot encoding (top‑N actors matter) ──
    df["actor_list"] = df["actors"].apply(lambda x: [a.strip() for a in x.split(",")][:3])
    mlb_actor = MultiLabelBinarizer()
    actor_encoded = pd.DataFrame(
        mlb_actor.fit_transform(df["actor_list"]),
        columns=[f"actor_{a}" for a in mlb_actor.classes_],
        index=df.index,
    )

    # ── Director encoding ──
    df["primary_director"] = df["directors"].apply(lambda x: x.split(",")[0].strip())
    le_director = LabelEncoder()
    df["director_encoded"] = le_director.fit_transform(df["primary_director"])

    # ── Synopsis sentiment ──
    if "synopsis" in df.columns:
        df["synopsis_score"] = df["synopsis"].apply(synopsis_score)
    else:
        df["synopsis_score"] = 0.0

    # ── Feature matrix ──
    X = pd.concat(
        [genre_encoded, actor_encoded, df[["director_encoded", "synopsis_score"]]],
        axis=1,
    )
    feature_cols = list(X.columns)
    y = df["rating"]

    # ── Train ──
    model = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    return model, df, mlb_genre, mlb_actor, le_director, feature_cols, genre_encoded, actor_encoded


model, df, mlb_genre, mlb_actor, le_director, feature_cols, genre_encoded, actor_encoded = load_and_train()

# Collect unique values for dropdowns
all_genres = sorted(set(g for gl in df["genre_list"] for g in gl))
all_directors = sorted(df["primary_director"].unique())


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🍿 About")
    st.markdown(
        """
        **CineMetric** uses a Random Forest Regressor trained on 
        real movie data to predict ratings based on genre, director, 
        cast, synopsis sentiment, and more!
        
        Fill in your movie's details and hit **Predict Rating** ✨
        """
    )

    st.markdown("---")
    st.markdown("## 🎀 Movie Details")

    movie_title = st.text_input("Movie Title", placeholder="e.g. The Last Horizon")

    selected_genre = st.selectbox(
        "Primary Genre",
        options=all_genres,
        index=all_genres.index("Drama") if "Drama" in all_genres else 0,
    )

    selected_director = st.selectbox(
        "Director",
        options=["— Custom (not in dataset) —"] + all_directors,
        index=0,
    )
    custom_director = None
    if selected_director == "— Custom (not in dataset) —":
        custom_director = st.text_input("Enter Director Name", placeholder="e.g. Jane Doe")

    actors_input = st.text_input(
        "Actors (comma-separated)",
        placeholder="e.g. Emma Stone, Ryan Gosling, John Cho",
    )

    synopsis_input = st.text_area(
        "Synopsis",
        placeholder="Briefly describe what the movie is about…",
        height=120,
    )

    budget = st.slider("Budget ($ Millions)", min_value=1, max_value=500, value=50, step=1)

    st.markdown("---")

    predict_clicked = st.button("✨  Predict Rating")


# ══════════════════════════════════════════════════════════════════════
# MAIN PAGE — HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🍿 CineMetric: AI Movie Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Predict movie ratings with a sprinkle of machine learning magic ✨</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════
def build_input_vector(genre, director_name, actors_str, synopsis_text):
    """Build a feature vector matching the training feature set."""
    warnings_list = []

    # ── Genre one-hot ──
    genre_vector = {col: 0 for col in genre_encoded.columns}
    genre_col = f"genre_{genre}"
    if genre_col in genre_vector:
        genre_vector[genre_col] = 1
    else:
        warnings_list.append(f"Genre **{genre}** not found in training data — using global average.")

    # ── Actor multi-hot (up to first 3 actors provided) ──
    actor_vector = {col: 0 for col in actor_encoded.columns}
    if actors_str and actors_str.strip():
        input_actors = [a.strip() for a in actors_str.split(",")][:3]
        matched = 0
        for actor in input_actors:
            col = f"actor_{actor}"
            if col in actor_vector:
                actor_vector[col] = 1
                matched += 1
        if matched == 0:
            warnings_list.append("None of the actors are in the training dataset — using global average contribution.")
    else:
        warnings_list.append("No actors provided — using global average actor contribution.")

    # ── Director encoding ──
    if director_name and director_name in le_director.classes_:
        dir_enc = le_director.transform([director_name])[0]
    else:
        dir_enc = int(np.median(df["director_encoded"]))
        label = director_name or "Unknown"
        warnings_list.append(f"Director **{label}** not in dataset — using global average.")

    # ── Synopsis sentiment ──
    syn_score = synopsis_score(synopsis_text)

    # ── Assemble ──
    row = (
        list(genre_vector.values())
        + list(actor_vector.values())
        + [dir_enc, syn_score]
    )
    return np.array(row).reshape(1, -1), warnings_list


if predict_clicked:
    if not movie_title.strip():
        st.warning("⚠️ Please enter a movie title in the sidebar.")
    else:
        director_name = custom_director if custom_director else (
            selected_director if selected_director != "— Custom (not in dataset) —" else None
        )

        input_vec, warn_msgs = build_input_vector(
            selected_genre, director_name, actors_input, synopsis_input
        )

        # ── Show warnings ──
        for w in warn_msgs:
            st.info(f"ℹ️ {w}")

        # ── Predict ──
        predicted_rating = model.predict(input_vec)[0]
        predicted_rating = round(np.clip(predicted_rating, 0, 10), 2)

        # ── Star visual ──
        filled = int(round(predicted_rating / 2))
        stars_html = "⭐" * filled + "🌑" * (5 - filled)

        # ── Display metric ──
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            emoji = "🌟" if predicted_rating >= 7 else ("👍" if predicted_rating >= 5 else "😬")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">✨ Predicted Rating ✨</div>
                    <div class="metric-value">{predicted_rating} / 10</div>
                    <div class="metric-sub">{emoji} {movie_title}</div>
                    <div class="star-badge">{stars_html}</div>
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
            fig1.patch.set_facecolor("#fff0f6")
            ax1.set_facecolor("#fce4ec")

            pastel_colors = ["#f48fb1", "#ce93d8", "#90caf9", "#80cbc4", "#ffcc80"]
            bars = ax1.barh(
                top5["title"],
                top5["rating"],
                color=pastel_colors[: len(top5)],
                edgecolor="white",
                linewidth=1.5,
                height=0.55,
            )
            for bar, rating in zip(bars, top5["rating"]):
                ax1.text(
                    bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{rating:.1f}",
                    va="center",
                    ha="left",
                    color="#7b4f72",
                    fontweight="bold",
                    fontsize=10,
                )

            ax1.set_xlim(0, 10.5)
            ax1.set_xlabel("Rating", color="#ad7da0", fontsize=10)
            ax1.tick_params(colors="#7b4f72", labelsize=9)
            ax1.invert_yaxis()
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["bottom"].set_color("#f8bbd0")
            ax1.spines["left"].set_color("#f8bbd0")
            ax1.grid(axis="x", color="#f8bbd0", linestyle="--", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        # ── Chart 2: Feature Importance ──
        with chart_col2:
            st.markdown("### 📊 Feature Importance")
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            top_imp = importances.nlargest(10).sort_values()

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_facecolor("#fff0f6")
            ax2.set_facecolor("#f3e5f5")

            gradient = plt.cm.RdPu(np.linspace(0.35, 0.85, len(top_imp)))
            ax2.barh(
                top_imp.index.str.replace("genre_", "").str.replace("actor_", "🎭 ").str.replace("_encoded", "").str.replace("synopsis_score", "📝 Synopsis"),
                top_imp.values,
                color=gradient,
                edgecolor="white",
                linewidth=1.2,
                height=0.55,
            )
            for i, (val, name) in enumerate(zip(top_imp.values, top_imp.index)):
                ax2.text(
                    val + 0.002,
                    i,
                    f"{val:.3f}",
                    va="center",
                    color="#7b4f72",
                    fontsize=9,
                    fontweight="bold",
                )

            ax2.set_xlabel("Importance", color="#ad7da0", fontsize=10)
            ax2.tick_params(colors="#7b4f72", labelsize=9)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["bottom"].set_color("#e1bee7")
            ax2.spines["left"].set_color("#e1bee7")
            ax2.grid(axis="x", color="#e1bee7", linestyle="--", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # ── Summary card ──
        actors_display = actors_input.strip() or "N/A"
        st.markdown("---")
        st.markdown(
            f"""
            <div class="info-card">
                <strong>🎬 Movie Summary</strong><br><br>
                <strong>Title:</strong> {movie_title} &nbsp;|&nbsp;
                <strong>Genre:</strong> {selected_genre} &nbsp;|&nbsp;
                <strong>Director:</strong> {director_name or 'N/A'} &nbsp;|&nbsp;
                <strong>Actors:</strong> {actors_display} &nbsp;|&nbsp;
                <strong>Budget:</strong> ${budget}M &nbsp;|&nbsp;
                <strong>Synopsis Sentiment:</strong> {synopsis_score(synopsis_input):+.2f} &nbsp;|&nbsp;
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
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">🍿</p>
                <p style="font-size: 1.15rem; color: #ad7da0; font-weight: 600;">
                    Fill in the movie details in the sidebar<br>
                    and click <strong style="color:#e91e8c;">Predict Rating ✨</strong> to get started.
                </p>
                <p style="font-size: 0.9rem; color: #c28fb1; margin-top: 1rem;">
                    Now with <strong>actor</strong> and <strong>synopsis</strong> analysis! 🎭📝
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#c28fb1; font-size:0.82rem;">'
    "CineMetric v2.0 — Built with 💖 Streamlit & Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
