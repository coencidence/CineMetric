# 🎬 CineMetric: AI Movie Predictor

**Live App:** [cinemetricmovies.streamlit.app](https://cinemetricmovies.streamlit.app)

CineMetric is a professional Python-based movie rating prediction system. It leverages machine learning to estimate movie ratings based on metadata such as genres, directors, and cast. The project includes both a development workflow in Jupyter Notebook and a polished, interactive web application built with Streamlit.

---

## 🚀 Features

- **Jupyter Notebook Workflow**: End-to-end ML pipeline from data cleaning to model evaluation.
- **Streamlit Web App**: A sleek, dark-themed dashboard for real-time predictions.
- **Advanced Feature Engineering**: 
  - **Multi-label Binarization** for handling movies with multiple genres.
  - **Label Encoding** for directors and lead actors.
- **Random Forest Prediction**: Uses a Random Forest Regressor for robust and interpretable score calculation.
- **Interactive Visualizations**:
  - **Top 5 Similar Movies**: Finds matching movies in the dataset.
  - **Feature Importance**: Visualizes what factors (genre, director, etc.) influenced the AI's score.
  - **Correlation Heatmap**: Analyzes relationships between features.
- **Graceful Error Handling**: Automatically uses global averages if a user inputs a director or genre not present in the training data.

---

## 🛠️ Tech Stack

- **Logic**: Python 3.11+
- **Data**: Pandas, NumPy
- **ML Model**: Scikit-Learn (Random Forest Regressor)
- **Web UI**: Streamlit
- **Visuals**: Matplotlib, Seaborn

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/coencidence/CineMetric.git
   cd CineMetric
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Launching the App

To start the interactive web application, run:

```bash
streamlit run app.py
```

The app will open in your default browser (usually at `http://localhost:8501`).

---

## 🧪 Development & Modeling

For a deep dive into the data analysis and model training process, refer to the [movie_rating_prediction.ipynb](movie_rating_prediction.ipynb) notebook. It covers:
- Data cleaning and deduplication.
- Handling of multiple genres per movie.
- Model performance metrics (MSE and R² Score).
- Distribution analysis of actual vs. predicted ratings.

---

## 📊 Dataset

The project uses `movies_data.csv`, which contains:
- `title`: Movie name
- `genres`: Comma-separated list of genres
- `synopsis`: Movie summary
- `rating`: IMDb/Target rating (0-10)
- `actors`: Cast list
- `directors`: Director(s) info

---
