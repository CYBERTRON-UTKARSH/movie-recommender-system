import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Movie Recommender", page_icon="🎬", layout="wide"
)


# Load CSV and compute similarity safely
@st.cache_resource
def load_data():
  try:
    # 📌 Change 'movies.csv' to your actual CSV filename if different
    movies = pd.read_csv("movies_metadata.csv")
    return movies, None
  except Exception as e:
    return None, str(e)


movies, error_message = load_data()

# Sidebar Layout
st.sidebar.markdown("### 🎬 Menu")
page = st.sidebar.radio("Go to", ["Home", "Search & Recommendations"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏠 Home Feed Settings")
category = st.sidebar.selectbox(
    "Category", ["trending", "popular", "top_rated"]
)
grid_columns = st.sidebar.slider(
    "Grid columns", min_value=2, max_value=6, value=6
)

# Main Header
st.markdown("### 🎬 Movie Recommender")
search_query = st.text_input(
    "Search by movie title (keyword)",
    placeholder="Type: avenger, batman, love...",
)

if movies is None:
  st.error(
      f"⚠️ Could not load your CSV file. Error: `{error_message}`. Make sure"
      " `movies_metadata.csv` is in the same folder as `app.py`."
  )
else:
  # Display available columns to help debug if needed
  # st.write("Columns in CSV:", list(movies.columns))


  @st.cache_resource
  def get_similarity(df):
    # Automatically pick the best text column
    possible_cols = ["tags", "overview", "genres", "description", "summary"]
    text_col = next(
        (col for col in possible_cols if col in df.columns), None
    )

    if text_col is None:
      # Fallback to the first string column found in the dataframe
      string_cols = df.select_dtypes(include=["object"]).columns
      if len(string_cols) > 1:
        text_col = string_cols[1]  # skip title if title is index 0
      else:
        text_col = df.columns[0]

    # Fill NaNs and ensure text type
    text_data = df[text_col].fillna("").astype(str)

    # Vectorize text
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(text_data).toarray()

    if vectors.shape[1] == 0:
      raise ValueError(
          f"Text column '{text_col}' resulted in 0 features. Please check your"
          " CSV format."
      )

    return cosine_similarity(vectors), text_col


  try:
    similarity, used_column = get_similarity(movies)
    # st.sidebar.success(f"Using text column: `{used_column}`")
  except Exception as e:
    st.error(f"Error building similarity matrix: {e}")
    similarity = None

  if similarity is not None:
    # Handle Search Results
    if search_query:
      st.markdown("---")
      st.subheader(f"🔍 Search Results for '{search_query}'")
      title_col = (
          "title"
          if "title" in movies.columns
          else movies.select_dtypes(include=["object"]).columns[0]
      )
      matched_movies = movies[
          movies[title_col].str.contains(search_query, case=False, na=False)
      ]

      if not matched_movies.empty:
        cols = st.columns(grid_columns)
        for idx, row in matched_movies.head(12).iterrows():
          col = cols[idx % grid_columns]
          with col:
            st.markdown(f"**{row[title_col]}**")
            if st.button("Get Recs", key=f"search_{idx}"):
              st.session_state["selected_movie"] = row[title_col]
      else:
        st.info("No movies found matching that keyword.")

    # Handle Selected Movie Recommendations
    if "selected_movie" in st.session_state:
      selected = st.session_state["selected_movie"]
      st.markdown("---")
      st.subheader(f"✨ Recommendations based on: *{selected}*")

      try:
        title_col = (
            "title"
            if "title" in movies.columns
            else movies.select_dtypes(include=["object"]).columns[0]
        )
        index = movies[movies[title_col] == selected].index[0]
        distances = sorted(
            enumerate(similarity[index]), reverse=True, key=lambda x: x[1]
        )

        rec_cols = st.columns(min(5, grid_columns))
        for i, item in enumerate(distances[1:6]):
          rec_idx = item[0]
          rec_title = movies.iloc[rec_idx][title_col]
          col = rec_cols[i % len(rec_cols)]
          with col:
            st.success(f"**{rec_title}**")
      except Exception as e:
        st.error(f"Error generating recommendations: {e}")

    # Home Feed Section
    if page == "Home":
      st.markdown("---")
      st.subheader(f"🏠 Home — {category.capitalize()}")

      title_col = (
          "title"
          if "title" in movies.columns
          else movies.select_dtypes(include=["object"]).columns[0]
      )
      feed_movies = movies.head(24)
      cols = st.columns(grid_columns)

      for idx, row in feed_movies.iterrows():
        col = cols[idx % grid_columns]
        with col:
          st.markdown(f"🎬 **{row[title_col]}**")
          if st.button("View Recs", key=f"home_{idx}"):
            st.session_state["selected_movie"] = row[title_col]
            st.rerun()