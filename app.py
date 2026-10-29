import os
import hashlib
from pathlib import Path
from urllib.parse import quote_plus

import requests

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TONIGHTSHOW | Movie Discovery",
    page_icon="🎬",
    layout="wide",
)

# ---------------- NEON STYLING ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(ellipse at top left, #241039, #090914 55%, #05060b);
        color: #f7f5ff;
        font-family: 'Space Grotesk', sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    .stDeployButton, #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden; }
    .block-container { max-width: 1450px; padding-top: 1rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #ffffff !important; }
    section[data-testid="stSidebar"] {
        background: #100c1d;
        border-right: 1px solid #302047;
    }
    .hero {
        padding: 36px;
        border-radius: 24px;
        border: 1px solid #35f5ff;
        background: linear-gradient(120deg, #111025, #28103a, #0b1020);
        box-shadow: 0 0 25px #35f5ff22;
        margin-bottom: 24px;
    }
    .brand {
        color: #35f5ff;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 4px;
        font-weight: 700;
    }
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(28px, 4vw, 48px);
        font-weight: 800;
        line-height: 1.25;
        margin: 16px 0;
    }
    .neon { color: #ff3bd4; text-shadow: 0 0 16px #ff3bd455; }
    .hero-copy { color: #d2cee6; font-size: 17px; }
    .metric {
        border: 1px solid #9d5cff66;
        background: #171226;
        padding: 16px;
        border-radius: 14px;
    }
    .metric-label { color: #aaa5c5; font-size: 12px; letter-spacing: 1px; }
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 25px;
        color: #35f5ff;
        margin-top: 6px;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #35f5ff;
        border-bottom-color: #ff3bd4;
    }
    .stButton button, .stFormSubmitButton button {
        border: 1px solid #35f5ff;
        border-radius: 9px;
        background: #171226;
        color: white;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        border-color: #ff3bd4;
        box-shadow: 0 0 12px #ff3bd455;
        color: white;
    }
    .movie-card {
        border: 1px solid #35f5ff55;
        border-radius: 16px;
        background: #11101e;
        padding: 12px;
        margin-bottom: 8px;
    }
    .poster-placeholder {
        height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 16px;
        background: linear-gradient(135deg, #171226, #28103a);
        border: 1px solid #35f5ff;
        border-radius: 12px;
        color: #35f5ff;
        font-family: 'Orbitron', sans-serif;
        font-size: 22px;
        font-weight: bold;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = "movie_recommendation_dataset_5000.csv"
FEEDBACK_FILE = "user_movie_ratings.csv"
POSTER_DIR = Path("posters")
POSTER_DIR.mkdir(exist_ok=True)
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def get_tmdb_token():
    """Read the TMDB token from Streamlit secrets or an environment variable."""
    try:
        token = st.secrets.get("TMDB_API_TOKEN", "")
        if token:
            return str(token).strip()
    except Exception:
        pass
    return os.getenv("TMDB_API_TOKEN", "").strip()


@st.cache_data(ttl=86400, show_spinner=False)
def search_tmdb_poster(title, year, token):
    """Search TMDB and return a poster URL when a match is found."""
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
    params = {"query": str(title), "include_adult": "false", "language": "en-US"}
    if year and int(year) > 0:
        params["year"] = int(year)
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            headers=headers, params=params, timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None
        title_norm = str(title).strip().casefold()
        exact = [r for r in results if str(r.get("title", "")).strip().casefold() == title_norm
                 or str(r.get("original_title", "")).strip().casefold() == title_norm]
        candidates = exact or results
        if year and int(year) > 0:
            same_year = [r for r in candidates if str(r.get("release_date", ""))[:4] == str(int(year))]
            if same_year:
                candidates = same_year
        poster_path = candidates[0].get("poster_path")
        return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    except (requests.RequestException, ValueError, TypeError):
        return None


def get_local_poster_path(title, year=0):
    """Download and cache a poster locally so it displays reliably in Streamlit."""
    safe_key = hashlib.sha1(f"{title}|{year}".encode("utf-8")).hexdigest()[:16]
    local_path = POSTER_DIR / f"{safe_key}.jpg"
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    token = get_tmdb_token()
    poster_url = search_tmdb_poster(title, year, token) if token else None
    if not poster_url:
        poster_url = POSTERS.get(title)
    if not poster_url:
        return None
    try:
        response = requests.get(poster_url, timeout=12)
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/") or not response.content:
            return None
        local_path.write_bytes(response.content)
        return local_path
    except (requests.RequestException, OSError):
        return None

# Optional external posters for a few well-known films.
# If an image URL fails or no poster exists, a styled local placeholder is shown.
POSTERS = {
    "Inception": "https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
    "Interstellar": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    "The Dark Knight": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "Coco": "https://image.tmdb.org/t/p/w500/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg",
    "The Godfather": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
    "Spirited Away": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
    "Your Name": "https://image.tmdb.org/t/p/w500/q719jXXEzOoYaps6dadT7wR7szk.jpg",
    "Dangal": "https://image.tmdb.org/t/p/w500/6ELJEzQJ3Y45HczMPIyeg6L0t6z.jpg",
    "Baahubali: The Beginning": "https://image.tmdb.org/t/p/w500/9BAjt8nSSms62uOVYn1t3C3dVto.jpg",
}

DESCRIPTIONS = {
    "Inception": "A skilled thief enters people's dreams to steal secrets and is offered a seemingly impossible final job.",
    "Interstellar": "Explorers travel beyond our galaxy in a desperate search for a future for humanity.",
    "The Dark Knight": "Batman faces a chaotic criminal mastermind threatening the people of Gotham.",
    "Coco": "A music-loving boy journeys into the Land of the Dead to uncover his family's history.",
    "The Godfather": "The aging head of a crime family prepares his reluctant son for a life-changing succession.",
    "Spirited Away": "A young girl enters a mysterious spirit world and must find a way to save her parents.",
    "Your Name": "Two teenagers mysteriously swap lives and try to find each other across time and distance.",
    "Dangal": "A former wrestler trains his daughters to compete against expectations and become champions.",
    "Baahubali: The Beginning": "A young man discovers his extraordinary heritage and becomes part of an epic kingdom struggle.",
    "3 Idiots": "Three engineering students question traditional education, ambition, friendship and success.",
    "Queen": "After her wedding is called off, a young woman travels alone and discovers her independence.",
    "Jersey": "A former cricketer returns to the sport to fulfil a promise to his son.",
}

# ---------------- HERO ----------------
st.markdown(
    """
    <div class="hero">
        <div class="brand">✦ TONIGHTSHOW</div>
        <div class="hero-title">
            YOUR NEXT FAVORITE<br>
            <span class="neon">IS WAITING.</span>
        </div>
        <div class="hero-copy">
            Discover movies tailored to your taste. Rate what you watch,
            and let the recommendation engine learn from your preferences.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- DATA CLEANING ----------------
@st.cache_data
def load_movies():
    if not os.path.exists(DATA_FILE):
        st.error(
            f"Dataset not found: {DATA_FILE}. "
            "Keep the CSV in the same folder as app.py."
        )
        st.stop()

    df = pd.read_csv(DATA_FILE)
    required = {"user_id", "rating", "movie_title", "genre", "language", "release_year"}
    missing = required - set(df.columns)
    if missing:
        st.error("Dataset is missing required columns: " + ", ".join(sorted(missing)))
        st.stop()

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["movie_title"] = df["movie_title"].astype("string").str.strip()
    df["genre"] = df["genre"].astype("string").fillna("Unknown")
    df["language"] = df["language"].astype("string").fillna("Unknown")

    df = df.dropna(subset=["user_id", "movie_title", "rating"])
    df = df[(df["movie_title"] != "") & df["rating"].between(1, 5)]
    df["user_id"] = df["user_id"].astype(int)
    df["release_year"] = df["release_year"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["user_id", "movie_title"], keep="last")
    return df


def build_recommendation_model(df):
    rating_matrix = df.pivot_table(
        index="user_id", columns="movie_title", values="rating", aggfunc="mean"
    )
    centered = rating_matrix.sub(rating_matrix.mean(axis=1), axis=0).fillna(0)
    similarity_matrix = pd.DataFrame(
        cosine_similarity(centered),
        index=rating_matrix.index,
        columns=rating_matrix.index,
    )
    return rating_matrix, similarity_matrix


def recommend_movies(user_id, rating_matrix, similarity_matrix, count=6):
    if user_id not in rating_matrix.index:
        return []

    user_ratings = rating_matrix.loc[user_id].dropna()
    already_rated = set(user_ratings.index)
    similar_users = (
        similarity_matrix.loc[user_id]
        .drop(index=user_id, errors="ignore")
        .sort_values(ascending=False)
    )
    similar_users = similar_users[similar_users > 0].head(40)

    weighted_scores = {}
    total_weights = {}
    for other_user, similarity_score in similar_users.items():
        for title, rating in rating_matrix.loc[other_user].dropna().items():
            if title in already_rated:
                continue
            weighted_scores[title] = weighted_scores.get(title, 0.0) + float(similarity_score) * float(rating)
            total_weights[title] = total_weights.get(title, 0.0) + abs(float(similarity_score))

    results = []
    for title, score in weighted_scores.items():
        weight = total_weights.get(title, 0)
        if weight > 0:
            results.append({
                "movie_title": title,
                "predicted_rating": max(1.0, min(5.0, score / weight)),
            })
    results.sort(key=lambda movie: movie["predicted_rating"], reverse=True)
    return results[:count]


def load_saved_ratings():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=["user_id", "movie_title", "rating", "updated_at"])
    try:
        saved = pd.read_csv(FEEDBACK_FILE)
        saved["user_id"] = pd.to_numeric(saved["user_id"], errors="coerce")
        saved["rating"] = pd.to_numeric(saved["rating"], errors="coerce")
        saved = saved.dropna(subset=["user_id", "movie_title", "rating"])
        return saved
    except Exception:
        return pd.DataFrame(columns=["user_id", "movie_title", "rating", "updated_at"])


def show_poster(title, year=0):
    """Show a locally cached poster or a styled placeholder."""
    local_poster = get_local_poster_path(title, year)
    if local_poster is not None and local_poster.exists():
        st.image(str(local_poster), use_container_width=True)
    else:
        safe_title = (str(title).replace("&", "&amp;").replace("<", "&lt;")
                      .replace(">", "&gt;"))
        st.markdown(
            f'<div class="poster-placeholder">🎬<br>{safe_title}</div>',
            unsafe_allow_html=True,
        )


movies_df = load_movies()
rating_matrix, similarity_matrix = build_recommendation_model(movies_df)
saved_ratings = load_saved_ratings()

# Apply saved feedback only to movie/profile pairs that exist in the dataset.
if not saved_ratings.empty:
    latest = saved_ratings.drop_duplicates(subset=["user_id", "movie_title"], keep="last")
    for _, row in latest.iterrows():
        uid = int(row["user_id"])
        title = str(row["movie_title"])
        if uid in rating_matrix.index and title in rating_matrix.columns:
            rating_matrix.loc[uid, title] = float(row["rating"])
    centered = rating_matrix.sub(rating_matrix.mean(axis=1), axis=0).fillna(0)
    similarity_matrix = pd.DataFrame(
        cosine_similarity(centered),
        index=rating_matrix.index,
        columns=rating_matrix.index,
    )

st.success("Movie dataset cleaned and recommendation model loaded.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ✦ TONIGHTSHOW")
    st.caption("YOUR PERSONAL MOVIE DISCOVERY ENGINE")
    user_ids = sorted(rating_matrix.index.tolist())
    if not user_ids:
        st.error("No viewer profiles were found in the dataset.")
        st.stop()
    selected_user = st.selectbox("Choose viewer profile", user_ids)
    recommendation_count = st.slider("Number of recommendations", 3, 9, 6)
    language_options = ["All languages"] + sorted(
        movies_df["language"].dropna().astype(str).unique().tolist()
    )
    selected_language = st.selectbox("Filter by language", language_options)

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)
for col, label, value in [
    (col1, "CLEAN RATINGS", len(movies_df)),
    (col2, "VIEWER PROFILES", len(rating_matrix.index)),
    (col3, "MOVIES", len(rating_matrix.columns)),
]:
    with col:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value:,}</div></div>',
            unsafe_allow_html=True,
        )

tab_recommendations, tab_rate, tab_history, tab_about = st.tabs(
    ["✦ FOR YOU", "★ RATE A MOVIE", "◷ YOUR RATINGS", "⌘ HOW IT WORKS"]
)

# ---------------- RECOMMENDATIONS ----------------
with tab_recommendations:
    st.subheader("Movies picked for your taste")
    results = recommend_movies(
        selected_user, rating_matrix, similarity_matrix, recommendation_count
    )
    movie_info = movies_df.drop_duplicates("movie_title").set_index("movie_title")

    if selected_language != "All languages":
        results = [
            item for item in results
            if item["movie_title"] in movie_info.index
            and str(movie_info.loc[item["movie_title"], "language"]) == selected_language
        ]

    if not results:
        st.warning("No recommendations available for this profile and filter. Try another profile or language.")
    else:
        for start in range(0, len(results), 3):
            card_columns = st.columns(3)
            for col, item in zip(card_columns, results[start:start + 3]):
                title = item["movie_title"]
                if title in movie_info.index:
                    info = movie_info.loc[title]
                    genre = str(info["genre"])
                    language = str(info["language"])
                    year = int(info["release_year"])
                else:
                    genre, language, year = "Unknown", "Unknown", 0

                description = DESCRIPTIONS.get(
                    title,
                    f"A {genre.replace('|', ', ').lower()} movie to explore based on your preferences.",
                )
                with col:
                    # Poster is shown before the details; use a local posters/<title>.jpg
                    # file for a guaranteed image. Otherwise a styled placeholder is shown
                    # for titles without a known external poster.
                    show_poster(title, year)
                    st.markdown(f"### {title}")
                    year_label = str(year) if year else "Year unavailable"
                    st.caption(f"{year_label} · {language} · {genre.replace('|', ' / ')}")
                    st.write(description)
                    st.markdown(
                        f'<div class="neon">✦ PREDICTED RATING: {item["predicted_rating"]:.1f}/5</div>',
                        unsafe_allow_html=True,
                    )
                    watch_url = "https://www.justwatch.com/in/search?q=" + quote_plus(title)
                    st.link_button("↗ Find where to watch", watch_url, use_container_width=True)
                    if st.button("★ Rate this movie", key=f"rate_{selected_user}_{title}", use_container_width=True):
                        st.session_state["movie_to_rate"] = title
                        st.info("Open the RATE A MOVIE tab to submit your rating.")

# ---------------- RATE A MOVIE ----------------
with tab_rate:
    st.subheader("Your opinion shapes your next recommendations")
    st.write(
        "Rate a movie from 1 to 5 stars. Your saved rating is used when rebuilding "
        "the recommendation profile. Feedback is stored locally in user_movie_ratings.csv."
    )
    all_titles = sorted(movies_df["movie_title"].dropna().astype(str).unique().tolist())
    default_title = st.session_state.get("movie_to_rate", all_titles[0] if all_titles else "")
    if all_titles:
        with st.form("submit_movie_rating"):
            selected_title = st.selectbox(
                "Movie title",
                all_titles,
                index=all_titles.index(default_title) if default_title in all_titles else 0,
            )
            user_rating = st.slider("Your rating", min_value=1, max_value=5, value=4)
            submitted = st.form_submit_button("✦ Save rating and update profile")
        if submitted:
            current = load_saved_ratings()
            if not current.empty:
                current = current[
                    ~((current["user_id"] == selected_user) & (current["movie_title"] == selected_title))
                ]
            new_rating = pd.DataFrame([{
                "user_id": selected_user,
                "movie_title": selected_title,
                "rating": user_rating,
                "updated_at": pd.Timestamp.now().isoformat(),
            }])
            pd.concat([current, new_rating], ignore_index=True).to_csv(FEEDBACK_FILE, index=False)
            st.success(f"Saved {user_rating}/5 for {selected_title}! Recommendations will refresh.")
            st.rerun()

# ---------------- RATING HISTORY ----------------
with tab_history:
    st.subheader(f"Saved ratings for viewer {selected_user}")
    own_ratings = (
        saved_ratings[saved_ratings["user_id"] == selected_user]
        if not saved_ratings.empty else pd.DataFrame()
    )
    if own_ratings.empty:
        st.info("No saved ratings yet. Rate a movie to start building your preference profile.")
    else:
        st.dataframe(
            own_ratings[["movie_title", "rating"]].rename(
                columns={"movie_title": "Movie", "rating": "Your rating"}
            ),
            use_container_width=True,
            hide_index=True,
        )

# ---------------- ABOUT ----------------
with tab_about:
    st.subheader("How TONIGHTSHOW recommends movies")
    st.markdown(
        """
        **1. Data cleaning**  
        Invalid ratings, missing identifiers, and duplicate user–movie records are removed.

        **2. User–movie matrix**  
        Each row represents a viewer, each column a movie, and each value a rating.

        **3. Cosine similarity**  
        The model compares rating patterns to find viewers with similar tastes.

        **4. Rating prediction**  
        Ratings from similar viewers are combined using similarity weights.

        **5. Feedback loop**  
        Your ratings are saved in `user_movie_ratings.csv` and used to refresh the model
        when the app reruns.

        **6. Streaming discovery**  
        Each movie links to a JustWatch India search. Check the streaming provider to confirm availability.

        **Movie posters:** With a TMDB Read Access Token configured in `.streamlit/secrets.toml`,
        TONIGHTSHOW searches TMDB using each movie title and release year, then downloads and caches
        the poster in the local `posters` folder. If no match is available, it uses a known poster
        where possible or displays a styled placeholder. This product uses the TMDB API but is not
        endorsed or certified by TMDB.
        """
    )

st.divider()
st.caption(
    "TONIGHTSHOW · AI/ML college prototype · "
    "The demo dataset is synthetic, so recommendations are illustrative. Movie posters powered by TMDB."
)
