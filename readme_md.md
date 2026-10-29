# 🎬 End-to-End Movie Recommender System

A lightweight, self-contained, and interactive **Movie Recommender System** built with **Python**, **Streamlit**, **Pandas**, and **Scikit-Learn**. This application computes content-based similarity on the fly from a movie dataset, providing instant recommendations, keyword search capabilities, and a dynamic home feed with zero external backend server dependencies.

---

## 🏗️ System Architecture

The application is structured as a monolithic local web application where the UI and the recommendation engine run concurrently inside Streamlit.

```text
┌────────────────────────────────────────────────────────┐
│                      Streamlit UI                      │
│   (Sidebar Controls, Search Bar, Grid Layout, State)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Data Processing & ML                 │
│  - Pandas (CSV Ingestion & Filtering)                  │
│  - CountVectorizer (Text Feature Extraction / BoW)     │
│  - Cosine Similarity (Vector Distance Matrix)          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                      Data Source                       │
│                     [ movies.csv ]                     │
└──────────────────────────┬─────────────────────────────┘
```

---

## ⚙️ Tech Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/) (Interactive widgets, custom column grids, session state management)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/) (Data cleaning, lookup, and filtering)
* **Machine Learning / NLP:** [Scikit-Learn](https://scikit-learn.org/) (`CountVectorizer`, `cosine_similarity`)

---

## 🔄 How It Works (Step-by-Step Process)

1. **Data Ingestion (`pandas.read_csv`)**:
   * The app reads the movie dataset from `movies.csv` upon startup.
   * It dynamically identifies the primary text column (such as `tags`, `overview`, or `genres`) used for content profiling.

2. **Text Vectorization (`CountVectorizer`)**:
   * Text data from the selected feature column is cleaned, converted to string, and tokenized.
   * `CountVectorizer` (with English stop-word removal and a maximum feature limit of 5,000) transforms the text descriptions into numerical word-count vectors (Bag-of-Words model).

3. **Similarity Matrix Computation (`cosine_similarity`)**:
   * Scikit-Learn computes the pairwise cosine similarity score between every single movie vector in the dataset.
   * This yields a square similarity matrix where each cell represents how closely related two movies are based on their text features.
   * The matrix and dataset are cached using `@st.cache_resource` for high performance across user interactions.

4. **User Interaction & Discovery**:
   * **Home Feed:** Displays a responsive grid of movies categorized by trending, popular, or top-rated options.
   * **Keyword Search:** Allows users to search for movies by typing keywords, instantly filtering titles using case-insensitive substring matching.

5. **Recommendation Engine**:
   * When a user selects a movie (either from the home feed or search results), the app retrieves its index in the dataset.
   * It extracts the corresponding row from the cosine similarity matrix and sorts all movies in descending order of similarity scores.
   * The system filters out the selected movie itself and displays the **Top 5 most similar recommended movies** in a clean grid layout.

---

## 📂 Project Structure

```text
mls/
│
├── app.py          # Main Streamlit application (UI + ML logic)
├── movies.csv      # Dataset containing movie titles and text features
└── README.md       # Project documentation
```

---

## 🚀 Installation & Running Locally

### 1. Prerequisites
Ensure you have Python installed (Python 3.8+ recommended).

### 2. Install Dependencies
Open your terminal inside the project folder and run:
```bash
pip install streamlit pandas scikit-learn
```

### 3. Add Your Dataset
Ensure your dataset file named `movies.csv` (containing at least a `title` column and a text feature column like `tags` or `overview`) is placed in the root directory alongside `app.py`.

### 4. Run the Streamlit App
Execute the following command in your terminal:
```bash
streamlit run app.py
```
This will automatically launch the app in your default web browser at `http://localhost:8501`.