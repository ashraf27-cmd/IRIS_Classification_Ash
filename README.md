# Iris Species Classification

## Project Overview

This project focuses on the classification of Iris flower species using Machine Learning techniques in Python. The Iris dataset is one of the most widely used datasets for introductory classification tasks and contains measurements of sepal length, sepal width, petal length, and petal width for three flower species:

* Setosa - 50 flowers
* Versicolor - 50 flowers
* Virginica - 50 flowers

The project demonstrates the complete Data Science workflow, including data exploration, visualization, feature analysis, model training, and performance evaluation.

## Technologies Used

# Python

* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn (ML workflow)

## Exploratory Data Analysis (EDA)

The dataset was analyzed using:

* Dataset inspection (`head()`, `info()`, `shape`)
* Missing value analysis
* Duplicate record detection
* Statistical summaries
* Boxplots for outlier detection
* Correlation heatmap
* Pairplot visualization

## Machine Learning Models

The following classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)

## Evaluation Metrics

Model performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

  * Precision
  * Recall
  * F1-Score

## Results

The models achieved high classification accuracy on the Iris dataset, demonstrating the effectiveness of supervised machine learning techniques for species prediction.

## Repo Structure
## Repo Structure

```text
IRIS_Classification_Ash
│
├── Iris Classification.py
├── README.md
├── LICENSE
├── .gitignore
│
└── Viz
    ├── corrmap_iris.png
    ├── iris_pairplot.png
    ├── petal_length_boxplot.png
    ├── petal_width_boxplot.png
    ├── sepal_length_boxplot.png
    └── sepal_width_boxplot.png
```
## Conclusion

The analysis showed that petal-related features are highly influential in distinguishing Iris species. Through exploratory data analysis and machine learning classification, the project successfully predicted flower species with excellent performance across multiple classification algorithms.
