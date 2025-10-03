# Simple User-Based Movie Recommendation System (ML-100K)

This mini project builds a beginner-friendly user-based collaborative filtering recommender using the MovieLens 100K dataset (files in the `ml-100k/` folder). It computes user similarity from a user–item rating matrix and recommends unseen movies with the highest weighted scores from similar users. The system is evaluated using Precision@K and compared against simple popularity baselines. Bonus sections add Item-Based CF and a simple Matrix Factorization (truncated SVD) model.

## Dataset
MovieLens 100K (provided locally):
- `u.data`: user-item ratings (user_id, item_id, rating, timestamp)
- `u.item`: movie metadata (title + genres)

Only `u.data` and `u.item` are used here.

## What the Notebook Does (Cell-by-Cell Outline)
1. Imports
2. Load ratings and movie metadata, merge titles
3. Explore rating distribution (bar chart)
4. Build user–item matrix (users x items)
5. Compute user–user cosine similarity
6. User-based CF recommendation function
7. Sample user-based recommendations
8. Per-user train/test split (~20% holdout)
9. Rebuild matrices on train set only
10. Evaluation-time recommender for user-based CF
11. Precision@K for user-based CF
12. Post-evaluation sample recommendations
13. Popularity baseline (mean rating ranking)
14. Popularity baseline Precision@K
15. Diagnostics for zero baseline precision and improved popularity variants
16. Popularity baseline with minimum rating count
17. Popularity baseline using frequency (most-rated items)
18. Build item–item similarity matrix (Item-Based CF)
19. Item-Based CF recommendation function
20. Precision@K for Item-Based CF
21. Truncated SVD (matrix factorization) reconstruction
22. SVD-based recommendation function
23. Precision@K for SVD recommender
24. Summary comparison (User CF vs Item CF vs SVD vs Popularity variants)

## Methods Implemented
### User-Based Collaborative Filtering
- Cosine similarity over zero-filled user rating vectors
- Weighted average of neighbor ratings for unseen items

### Popularity Baselines
- Raw mean rating ranking (may include sparse, inflated items)
- Mean rating with minimum count filter
- Frequency (most-rated) ranking

### Item-Based Collaborative Filtering (Bonus)
- Cosine similarity between item rating vectors (items x users)
- For each candidate item, aggregate the target user's ratings of its most similar rated items
- Adjustable number of neighbor items (`top_n_neighbors`)

### Matrix Factorization via SVD (Bonus)
- Truncated SVD (numpy.linalg.svd) on user–item matrix with chosen rank (default 50)
- Reconstructed dense matrix used for scoring unseen items
- No regularization or bias terms for simplicity

## Evaluation: Precision@K
Precision@K = relevant_recommended / K
- Relevance threshold: rating >= 4 in held-out test ratings
- Users with no relevant test items are skipped
- Reported score is mean over remaining users

## Comparison (Typical Expectations)
- User-Based CF: Captures taste similarity; may be sensitive to sparsity
- Item-Based CF: Often more stable when users have rated enough items
- SVD: Can generalize and smooth noise; performance depends on rank choice
- Popularity (frequency): Strong baseline for coverage
- Popularity (raw mean): Poor without support filtering due to single-rating inflation

## How to Run
Install dependencies:
```
pip install -r requirements.txt
```
Open the notebook `loan_approval_prediction_description.ipynb` and run cells in order.

## Simplifications / Learning-Focused Choices
- Zero fill instead of mean-centering or normalization
- No similarity shrinkage or weighting by co-rated count
- SVD without biases or regularization
- Single metric (Precision@K) for clarity

## Possible Next Improvements
- Normalize ratings (subtract user mean) before similarity or SVD
- Add user/item bias terms (baseline predictors) to SVD
- Introduce regularized matrix factorization (ALS, SGD) or implicit feedback models
- Add Recall@K, MAP, NDCG, coverage, diversity metrics
- Hybrid scoring (blend CF + popularity)
- Parameter sweeps for rank, neighbor counts, min similarity thresholds

## Requirements
See `requirements.txt` for the minimal libraries used.

## License
See `LICENSE` in repository (if provided).

---
This README reflects the extended state of the learning-oriented recommender notebook including User-Based CF, Item-Based CF, SVD, multiple baselines, and comparative evaluation.
