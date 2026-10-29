# CineMatch — Personalized Movie Recommendation System

A small Streamlit demonstration of user-based collaborative filtering using cosine similarity.

## Files
- `app.py`: interactive app and recommendation model
- `movie_recommendation_dataset_5000.csv`: required dataset (copy it into this folder)
- `requirements.txt`: Python dependencies

## Run on Windows
1. Install Python 3.10+ if not already installed.
2. Open Command Prompt in this folder.
3. Create an environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Copy `movie_recommendation_dataset_5000.csv` into this folder.
6. Start the app:
   ```bash
   streamlit run app.py
   ```

The app opens in your browser. Select a user in the sidebar and show recommendations, rating history, and the model explanation.

## Important data note
The provided 5,000-row dataset is synthetic and includes intentionally planted anomalies. The app cleans common anomalies before creating the user–movie matrix. For a real-world claim, use a public dataset such as MovieLens and evaluate with a held-out test set.
