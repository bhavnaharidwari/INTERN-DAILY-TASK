# SalaryInsight - Clustering Tasks

Run in order (all files need `employee_data.csv` in the same folder):

    pip install pandas numpy scikit-learn scipy matplotlib
    python task1_kmeans_clustering.py
    python task2_optimal_clusters.py
    python task3_cluster_analysis.py

## Task 1 - k-Means with different distance metrics
- Features: Performance_Rating, Weekly_Hours, Training_Hours, Certifications, Work_Life_Balance (scaled).
- Missing values (25 each in Performance_Rating and Training_Hours) filled with the median.
- Euclidean and cosine give almost the same groups (ARI 0.96). Manhattan differs (ARI 0.35).
- A single random start is NOT stable (avg ARI ~0.28). Using the best of 10 starts makes
  euclidean (0.97) and cosine (0.96) stable.

## Task 2 - Optimal k
- Silhouette is almost flat (0.147 - 0.155 for k = 2..10), so the data has weak, overlapping clusters.
- The elbow method (distance-from-line) points to k = 5, and silhouette at k = 5 (0.149)
  is within 0.006 of the best score. Final k = 5.
- Cluster 3 is the weakest (avg silhouette 0.09, 68 negative values).

## Task 3 - Cluster analysis (k = 5)
Key distinguishing factors (ANOVA): Training_Hours > Certifications > Work_Life_Balance > Weekly_Hours > Performance_Rating.
Age and Experience do not differ between clusters.

| Cluster | Size | Description | Engagement |
|---|---|---|---|
| 0 | 486 | Heavy trainers, long hours, good work-life balance | Highest (0.59) |
| 1 | 377 | Low performance (2.78), long hours, poor balance, little training | Lowest (0.26), only 6% highly satisfied |
| 2 | 480 | Best performers (3.69), fewer hours, best balance, little training | Good (0.46), 42% highly satisfied |
| 3 | 276 | Certification-focused group (avg 3.1 certifications) | Medium (0.49) |
| 4 | 381 | Short hours, lots of training, poor balance | Medium-low (0.42) |

Productivity trends:
- Cluster 1 works the most hours but has the lowest output (burnout risk).
- Clusters 2 and 4 get the highest rating per hour worked.
- Performance does not rise with experience, training hours or certifications (correlations near 0).
- Salary has only a weak link with performance (corr 0.20), and cluster salaries are close (22.2 - 25.1 LPA);
  cluster 1 is lowest.
- Departments are spread evenly across clusters.

Note: the engagement score is my own simple formula (work-life balance + training hours + satisfaction, each scaled 0-1).
