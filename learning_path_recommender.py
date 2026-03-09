import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


class LearningPathRecommender:
    """
    Learning Path Recommendation System using Collaborative Filtering (Matrix Factorization).

    - Uses intern–course interaction data (ratings / completion scores)
    - Uses course metadata to present more informative recommendations
    """

    def __init__(
        self,
        n_components: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state

        # Data frames
        self.interactions_df: Optional[pd.DataFrame] = None
        self.courses_df: Optional[pd.DataFrame] = None

        # Encoders (mapping ids to indices)
        self.intern_index_: Optional[pd.Index] = None
        self.course_index_: Optional[pd.Index] = None

        # Matrix factorization model
        self.svd: Optional[TruncatedSVD] = None
        self.intern_factors_: Optional[np.ndarray] = None
        self.course_factors_: Optional[np.ndarray] = None

    def load_data(
        self,
        interactions_path: str = "intern_course_interactions.csv",
        courses_path: str = "courses_metadata.csv",
    ) -> None:
        """
        Load interaction and course metadata CSVs.

        Expected columns:
        - interactions: intern_id, course_id, rating
        - courses: course_id, title, category, difficulty, duration_hours
        """
        if not os.path.exists(interactions_path):
            raise FileNotFoundError(
                f"Interactions file not found: {interactions_path}. "
                "Create 'intern_course_interactions.csv' with columns: "
                "intern_id, course_id, rating."
            )

        if not os.path.exists(courses_path):
            raise FileNotFoundError(
                f"Courses file not found: {courses_path}. "
                "Create 'courses_metadata.csv' with columns: "
                "course_id, title, category, difficulty, duration_hours."
            )

        self.interactions_df = pd.read_csv(interactions_path)
        self.courses_df = pd.read_csv(courses_path)

        required_inter_cols = {"intern_id", "course_id", "rating"}
        required_course_cols = {"course_id", "title"}

        if not required_inter_cols.issubset(self.interactions_df.columns):
            raise ValueError(
                f"interactions file must contain columns: {required_inter_cols}"
            )
        if not required_course_cols.issubset(self.courses_df.columns):
            raise ValueError(
                f"courses file must contain at least columns: {required_course_cols}"
            )

    def _build_interaction_matrix(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build intern–course interaction matrix from interactions_df.

        Returns:
            pivot_df: DataFrame of shape (n_interns, n_courses)
            interaction_matrix: np.ndarray with NaNs filled as 0.0
        """
        if self.interactions_df is None:
            raise RuntimeError("Interactions data not loaded. Call load_data() first.")

        pivot_df = self.interactions_df.pivot_table(
            index="intern_id",
            columns="course_id",
            values="rating",
            aggfunc="mean",
        )

        # Save indices for mapping later
        self.intern_index_ = pivot_df.index
        self.course_index_ = pivot_df.columns

        # Replace NaN with 0 for SVD
        interaction_matrix = pivot_df.fillna(0.0).to_numpy(dtype=float)

        return pivot_df, interaction_matrix

    def fit(self) -> None:
        """
        Train a matrix factorization model (Truncated SVD) on the interaction matrix.
        """
        _, interaction_matrix = self._build_interaction_matrix()

        n_components = min(self.n_components, min(interaction_matrix.shape) - 1)

        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=self.random_state,
        )
        self.intern_factors_ = self.svd.fit_transform(interaction_matrix)
        self.course_factors_ = self.svd.components_.T

    def _predict_scores_for_all_courses(self) -> np.ndarray:
        """
        Reconstruct approximate rating matrix from latent factors.

        Returns:
            score_matrix of shape (n_interns, n_courses)
        """
        if self.intern_factors_ is None or self.course_factors_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        score_matrix = np.dot(self.intern_factors_, self.course_factors_.T)

        scaler = MinMaxScaler()
        score_matrix_scaled = scaler.fit_transform(score_matrix)

        return score_matrix_scaled

    def recommend_for_intern(
        self,
        intern_id,
        top_n: int = 5,
        filter_by_difficulty: Optional[str] = None,
        preferred_category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Recommend top-N courses for a given intern.
        """
        if self.interactions_df is None or self.courses_df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        if self.intern_index_ is None or self.course_index_ is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        if intern_id not in self.intern_index_:
            raise ValueError(f"Unknown intern_id: {intern_id}")

        intern_pos = self.intern_index_.get_loc(intern_id)
        score_matrix = self._predict_scores_for_all_courses()

        intern_scores = score_matrix[intern_pos, :]

        seen_courses = self.interactions_df.loc[
            self.interactions_df["intern_id"] == intern_id, "course_id"
        ].unique()

        recommendations = []
        for col_idx, course_id in enumerate(self.course_index_):
            if course_id in seen_courses:
                continue
            score = intern_scores[col_idx]
            course_row = self.courses_df[self.courses_df["course_id"] == course_id]
            if course_row.empty:
                continue

            course_row = course_row.iloc[0]
            rec = {
                "intern_id": intern_id,
                "course_id": course_id,
                "predicted_score": float(score),
                "title": course_row.get("title", ""),
                "category": course_row.get("category", ""),
                "difficulty": course_row.get("difficulty", ""),
                "duration_hours": course_row.get("duration_hours", np.nan),
            }
            recommendations.append(rec)

        rec_df = pd.DataFrame(recommendations)

        if rec_df.empty:
            return rec_df

        if filter_by_difficulty:
            rec_df = rec_df[
                rec_df["difficulty"].astype(str).str.lower()
                == filter_by_difficulty.lower()
            ]

        if preferred_category:
            rec_df = rec_df[
                rec_df["category"].astype(str).str.lower()
                == preferred_category.lower()
            ]

        rec_df = rec_df.sort_values(by="predicted_score", ascending=False)
        return rec_df.head(top_n).reset_index(drop=True)

    def build_learning_path(
        self,
        intern_id,
        top_n: int = 10,
        strategy: str = "difficulty_ascending",
    ) -> pd.DataFrame:
        """
        Build an ordered learning path for a given intern.
        """
        rec_df = self.recommend_for_intern(intern_id=intern_id, top_n=top_n * 3)

        if rec_df.empty:
            return rec_df

        if strategy == "difficulty_ascending":
            difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2}

            def diff_rank(x):
                return difficulty_order.get(str(x).lower(), 1)

            rec_df["difficulty_rank"] = rec_df["difficulty"].apply(diff_rank)
            rec_df = rec_df.sort_values(
                by=["difficulty_rank", "predicted_score"], ascending=[True, False]
            )
            rec_df = rec_df.drop(columns=["difficulty_rank"])

        elif strategy == "shortest_first":
            rec_df = rec_df.sort_values(
                by=["duration_hours", "predicted_score"], ascending=[True, False]
            )

        else:
            raise ValueError(
                "Unknown strategy. Use 'difficulty_ascending' or 'shortest_first'."
            )

        return rec_df.head(top_n).reset_index(drop=True)


def main() -> None:
    """
    Example usage.
    """
    recommender = LearningPathRecommender(n_components=20, random_state=42)

    recommender.load_data(
        interactions_path="intern_course_interactions.csv",
        courses_path="courses_metadata.csv",
    )

    recommender.fit()

    sample_intern_id = recommender.interactions_df["intern_id"].iloc[0]

    print(f"Building learning path for intern: {sample_intern_id}")
    learning_path = recommender.build_learning_path(
        intern_id=sample_intern_id,
        top_n=5,
        strategy="difficulty_ascending",
    )

    if learning_path.empty:
        print("No recommendations available (maybe the intern has seen all courses).")
    else:
        print("\nRecommended learning path:")
        for idx, row in learning_path.iterrows():
            print(
                f"{idx + 1}. {row['title']} "
                f"(course_id={row['course_id']}, "
                f"category={row.get('category', '')}, "
                f"difficulty={row.get('difficulty', '')}, "
                f"duration_hours={row.get('duration_hours', '')}, "
                f"predicted_score={row['predicted_score']:.3f})"
            )


if __name__ == "__main__":
    main()

