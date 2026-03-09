import os

import pandas as pd
import streamlit as st

from learning_path_recommender import LearningPathRecommender


def load_data(interactions_path: str, courses_path: str):
    interactions_df = pd.read_csv(interactions_path)
    courses_df = pd.read_csv(courses_path)
    return interactions_df, courses_df


def init_recommender(n_components: int):
    return LearningPathRecommender(n_components=n_components, random_state=42)


def main():
    st.title("Learning Path Recommendation System")
    st.write(
        "Personalized learning path recommendations for interns "
        "using Collaborative Filtering (Matrix Factorization)."
    )

    st.sidebar.header("Configuration")
    interactions_path = st.sidebar.text_input(
        "Interactions CSV path",
        value="intern_course_interactions.csv",
        help="File with columns: intern_id, course_id, rating",
    )
    courses_path = st.sidebar.text_input(
        "Courses metadata CSV path",
        value="courses_metadata.csv",
        help="File with columns: course_id, title, category, difficulty, duration_hours",
    )

    n_components = st.sidebar.slider(
        "Number of latent factors (n_components)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    top_n = st.sidebar.slider(
        "Top N recommendations in path",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
    )

    strategy = st.sidebar.selectbox(
        "Learning path strategy",
        options=["difficulty_ascending", "shortest_first"],
        index=0,
        help=(
            "'difficulty_ascending': start from easier courses.\n"
            "'shortest_first': sort by duration_hours then score."
        ),
    )

    st.sidebar.markdown("---")
    filter_difficulty = st.sidebar.selectbox(
        "Filter by difficulty (optional)",
        options=["", "Beginner", "Intermediate", "Advanced"],
        index=0,
    )
    preferred_category = st.sidebar.text_input(
        "Preferred category (optional)",
        value="",
    )

    st.markdown("### 1. Load data and train model")

    if st.button("Load data & train model"):
        try:
            if not os.path.exists(interactions_path):
                st.error(f"Interactions file not found: {interactions_path}")
                return
            if not os.path.exists(courses_path):
                st.error(f"Courses file not found: {courses_path}")
                return

            interactions_df, courses_df = load_data(
                interactions_path, courses_path
            )
            st.success(
                f"Loaded {len(interactions_df)} interactions and "
                f"{len(courses_df)} courses."
            )

            recommender = init_recommender(n_components=n_components)
            recommender.interactions_df = interactions_df
            recommender.courses_df = courses_df
            recommender.fit()

            st.session_state["recommender"] = recommender
            st.session_state["intern_ids"] = list(
                recommender.interactions_df["intern_id"].unique()
            )

            st.success("Model trained successfully.")
        except Exception as e:
            st.error(f"Error while loading data or training model: {e}")

    if "recommender" not in st.session_state:
        st.info("Train the model first, then you can request recommendations.")
        return

    recommender: LearningPathRecommender = st.session_state["recommender"]
    intern_ids = st.session_state.get("intern_ids", [])

    st.markdown("### 2. Get learning path for an intern")

    if not intern_ids:
        st.warning("No intern IDs found in interactions data.")
        return

    selected_intern = st.selectbox(
        "Select intern ID",
        options=intern_ids,
    )

    if st.button("Generate Learning Path"):
        try:
            difficulty_filter_value = (
                filter_difficulty if filter_difficulty.strip() != "" else None
            )
            preferred_category_value = (
                preferred_category if preferred_category.strip() != "" else None
            )

            learning_path = recommender.build_learning_path(
                intern_id=selected_intern,
                top_n=top_n,
                strategy=strategy,
            )

            if difficulty_filter_value or preferred_category_value:
                learning_path = learning_path.copy()
                if difficulty_filter_value:
                    learning_path = learning_path[
                        learning_path["difficulty"]
                        .astype(str)
                        .str.lower()
                        == difficulty_filter_value.lower()
                    ]
                if preferred_category_value:
                    learning_path = learning_path[
                        learning_path["category"]
                        .astype(str)
                        .str.lower()
                        == preferred_category_value.lower()
                    ]

            if learning_path.empty:
                st.warning(
                    "No recommendations available for this intern "
                    "(maybe they have seen all courses or filters are too strict)."
                )
            else:
                st.success("Learning path generated.")
                st.dataframe(learning_path)

                st.markdown("#### Ordered learning path")
                for idx, row in learning_path.iterrows():
                    st.write(
                        f"{idx + 1}. **{row['title']}** "
                        f"(course_id={row['course_id']}, "
                        f"category={row.get('category', '')}, "
                        f"difficulty={row.get('difficulty', '')}, "
                        f"duration_hours={row.get('duration_hours', '')}, "
                        f"predicted_score={row['predicted_score']:.3f})"
                    )
        except Exception as e:
            st.error(f"Error while generating learning path: {e}")


if __name__ == "__main__":
    main()

