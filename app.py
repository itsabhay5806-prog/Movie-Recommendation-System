
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET = "tmdb_5000_movies.csv"


def extract_names(value):
    """Extract names from TMDB JSON-like columns."""
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        return " ".join(
            item.get("name", "")
            for item in items
            if isinstance(item, dict)
        )
    except (ValueError, SyntaxError, TypeError):
        return ""


def load_movies():
    df = pd.read_csv(DATASET)

    required = ["title", "genres", "keywords", "overview"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df.dropna(subset=["title"]).copy()
    df = df.drop_duplicates(subset=["title"])
    df["overview"] = df["overview"].fillna("")

    # Convert genre and keyword metadata into readable text.
    df["genres_text"] = df["genres"].apply(extract_names)
    df["keywords_text"] = df["keywords"].apply(extract_names)

    # Combine features used to compare movies.
    df["features"] = (
        df["genres_text"] + " "
        + df["keywords_text"] + " "
        + df["overview"]
    ).str.lower()

    df = df.reset_index(drop=True)
    return df


def build_recommender(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    matrix = vectorizer.fit_transform(df["features"])
    similarity = cosine_similarity(matrix)

    return similarity


def recommend_movies(movie_title, df, similarity, top_n=5):
    matches = df.index[
        df["title"].str.casefold() == movie_title.strip().casefold()
    ].tolist()

    if not matches:
        print(f"\nMovie '{movie_title}' was not found.")
        suggestions = df[
            df["title"].str.contains(
                movie_title.strip(), case=False, na=False, regex=False
            )
        ]["title"].head(10).tolist()

        if suggestions:
            print("Did you mean:")
            for title in suggestions:
                print(f"  - {title}")
        else:
            print("Try another movie title.")

        return

    movie_index = matches[0]

    scores = list(enumerate(similarity[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    # Skip the selected movie itself.
    recommendations = [
        (index, score)
        for index, score in scores
        if index != movie_index
    ][:top_n]

    print(f"\nMovies similar to '{df.iloc[movie_index]['title']}':\n")

    for rank, (index, score) in enumerate(recommendations, start=1):
        movie = df.iloc[index]
        print(f"{rank}. {movie['title']}")
        print(f"   Genres: {movie['genres_text'] or 'Unknown'}")
        print(f"   Similarity: {score:.2%}")
        print(f"   Overview: {movie['overview'][:180]}")
        print()


def main():
    try:
        print("Loading movie dataset...")
        movies = load_movies()

        print("Building recommendation model...")
        similarity = build_recommender(movies)

        print(f"\nLoaded {len(movies)} movies successfully.")
        print("Type 'exit' to stop.")

        while True:
            title = input("\nEnter a movie title: ").strip()

            if title.lower() == "exit":
                print("Goodbye!")
                break

            if not title:
                print("Please enter a movie title.")
                continue

            recommend_movies(title, movies, similarity)

    except FileNotFoundError:
        print(
            f"Dataset not found: {DATASET}\n"
            "Download the TMDB dataset and place the CSV "
            "in the same folder as app.py."
        )
    except ValueError as error:
        print(f"Dataset error: {error}")


if __name__ == "__main__":
    main()
