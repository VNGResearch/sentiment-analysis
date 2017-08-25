# Unsupervised cross-domain sentiment analysis

## Goal
Create model for sentiment analysis based on unsupervised clustering and boosted learning.

## Approaches
- Construct an engineered set of feature words for sentiment analysis. Possible methods: bag-of-words univariate feature selection, logistic regression (using trained weights).
- Using several sample input sentences for each type of sentiment, define the corresponding initial K-means centroids as the centroids of each input clusters. This helps constraining the clustering to the specified domain, while also making it easier to infer cluster labels and estimating quality.
