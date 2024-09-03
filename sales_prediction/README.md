# Sales Prediction Using Logistic Regression and KNN

This project uses Logistic Regression and K-Nearest Neighbors (KNN) algorithms to predict whether a customer will buy a product based on their age and estimated salary.

## Project Structure

- `data/`: Contains the dataset used for the project.
- `src/`: Contains Python scripts for training and evaluating models.
- `README.md`: Overview of the project.
- `roc_curve.png`: ROC curve for Logistic Regression.
- `k_value_graph.png`: graph to find best k value.

## Dataset

The dataset is included in the `data/` folder and contains the following columns:
- `Age`: The age of the customer.
- `EstimatedSalary`: The estimated salary of the customer.
- `Purchased`: Whether the customer purchased the product (1 = Yes, 0 = No).

## Results

1. Logistic Regression
Accuracy: 80.0%
Confusion Matrix:
[[61,  0],
 [20, 19]]
ROC-AUC Score: 0.93
ROC Curve

2. KNN
Accuracy: 39.0%
Confusion Matrix:
[[ 0, 61],
 [ 0, 39]]
k- value graph 

## Conclusion
Logistic Regression performed better with an accuracy of 80% and an AUC score of 0.93.
KNN did not perform well, with an accuracy of 39%. The choice of k and the nature of the dataset may have influenced this result.
