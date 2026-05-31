# Hybrid Movie Recommendation Engine

A Python-based Data Science and Behavioral Modeling project that builds a Hybrid Recommendation System using the MovieLens-100k dataset. The engine merges User-Behavior Collaborative Filtering with Metadata Content-Based Filtering to resolve standard popularity biases.

---

## 📊 Core Architecture & Logic

The engine avoids spurious, random correlations (e.g., matching niche classic cartoons with gritty crime dramas) by passing user input through a dual-layered sorting pipeline:

1. **Collaborative Layer:** Builds a sparse $943 \times 1664$ User-Item Pivot Matrix. It evaluates the linear vector relationship of rating arrays across users via **Pearson Correlation Coefficients** using `pandas.corrwith()`.
2. **Content-Based Layer:** Identifies target genre flags across 19 categories. It dynamically suppresses candidate recommendations that share zero common genres with the baseline movie, enforcing genre cohesion.
3. **Popularity Thresholding:** Enforces a minimum review volume constraint to eliminate statistical anomalies caused by high reviews on obscure movies.

---

## 📈 System Performance Demonstration

When queried with the 1997 Noir-Thriller **'L.A. Confidential'**, the engine filters out irrelevant high-rating vectors and returns contextualized genre matches:

* **Target Profiles Identified:** `['Crime', 'Film-Noir', 'Mystery', 'Thriller']`
* **Top Recommends Served:**
  * *The Grifters (1990)* — Correlation: 0.54
  * *Breakdown (1997)* — Correlation: 0.48
  * *Taxi Driver (1976)* — Correlation: 0.44

---

## 🚀 How to Execute

1. Clone or download the `.ipynb` notebook file.
2. Launch the script inside a Jupyter environment or Google Colab.
3. Execute the single unified cell pipeline. It will automatically download the remote dataset, reconstruct the statistical indices, suppress sparse matrices warnings, and evaluate the query outputs instantly.# Movie-Recommendation-System-Hybrid
A Hybrid Recommendation Engine using Collaborative Filtering (User-Item Matrix Correlation) and Content-Based Metadata Filtering in PyTorch/Pandas
