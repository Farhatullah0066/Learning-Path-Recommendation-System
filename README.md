# Learning Path Recommendation System (Assignment 3)

This project implements a simple **Learning Path Recommendation System** for interns using **Collaborative Filtering (Matrix Factorization)**.

The goal is to recommend **personalized learning modules** (courses) for each intern based on:
- **Past intern learning patterns** (which courses they took and how they performed/liked them)
- **Course content metadata** (category, difficulty level, estimated duration, etc.)

## Files

- `learning_path_recommender.py`: Main Python script that:
  - Loads interaction data and course metadata from CSV files
  - Trains a matrix factorization model using collaborative filtering
  - Generates top-N learning module recommendations for each intern
  - Builds an ordered learning path for a given intern

- `requirements.txt`: Python dependencies for this assignment.

## Expected Data (CSV Files)

You can create these CSV files in the same folder (`ML_Project_3`):

1. `intern_course_interactions.csv`
   - Columns:
     - `intern_id` (string or int)
     - `course_id` (string or int)
     - `rating` (numeric, e.g. 1–5 or completion score)
   - Each row describes **how much an intern liked or completed a course**.

2. `courses_metadata.csv`
   - Columns:
     - `course_id`
     - `title`
     - `category` (e.g. "Python", "Machine Learning", "Data Science")
     - `difficulty` (e.g. "Beginner", "Intermediate", "Advanced")
     - `duration_hours` (numeric)

## How the Model Works

- Uses **Collaborative Filtering**:
  - Builds an **intern–course interaction matrix** from `intern_course_interactions.csv`
  - Factorizes it into **Intern latent vectors** and **Course latent vectors**
  - Predicts missing interaction values (how much an intern would like a course they have not taken yet)
- Recommendations:
  - For a given intern, sort all unseen courses by predicted score
  - Optionally filter/sort by metadata (difficulty, duration, category) to form a **learning path**

## Running the Script

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your CSV files (`intern_course_interactions.csv` and `courses_metadata.csv`) are in the same folder.

4. Run the recommender:

```bash
python learning_path_recommender.py
```

The script contains an example `if __name__ == "__main__":` block that:
- Loads data
- Trains the model
- Prints top-N recommended courses (learning path) for a sample intern.

You can open and edit all these files easily in **VS Code**.

