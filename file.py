# Data Analysis and Visualization Project
# Using the Iris dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

## Task 1: Load and Explore the Dataset

# Load the Iris dataset
try:
    iris = load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['target'])
    
    # Map target values to species names
    iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    display(iris_df.head())
    
    # Explore dataset structure
    print("\nDataset information:")
    iris_df.info()
    
    # Check for missing values
    print("\nMissing values per column:")
    print(iris_df.isnull().sum())
    
    # Clean dataset (though Iris typically has no missing values)
    # This is just for demonstration
    iris_df_clean = iris_df.dropna()  # Would drop rows with missing values if any
    
    print("\nDataset shape before and after cleaning (though no changes expected for Iris):")
    print(f"Before: {iris_df.shape}, After: {iris_df_clean.shape}")
    
except Exception as e:
    print(f"Error loading or processing dataset: {e}")

## Task 2: Basic Data Analysis

# Basic statistics
print("\nBasic statistics for numerical columns:")
display(iris_df_clean.describe())

# Group by species and compute mean of numerical columns
print("\nMean values by species:")
species_stats = iris_df_clean.groupby('species').mean()
display(species_stats)

# Interesting findings
print("\nInteresting findings:")
print("- Setosa has significantly smaller petal dimensions compared to other species")
print("- Virginica has the largest sepal length on average")
print("- Versicolor is in between setosa and virginica for most measurements")

## Task 3: Data Visualization

# Since Iris doesn't have time-series data, we'll modify the line chart requirement to show measurements by species

# 1. Line chart (modified to show feature means by species)
plt.figure(figsize=(12, 6))
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for feature in features:
    plt.plot(species_stats.index, species_stats[feature], marker='o', label=feature)
plt.title('Average Measurements by Iris Species')
plt.xlabel('Species')
plt.ylabel('Measurement (cm)')
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=iris_df_clean, estimator=np.mean, ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram (distribution of sepal length)
plt.figure(figsize=(8, 6))
sns.histplot(data=iris_df_clean, x='sepal length (cm)', hue='species', kde=True)
plt.title('Distribution of Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Count')
plt.show()

# 4. Scatter plot (sepal length vs petal length)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df_clean, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

# Additional visualization: Pairplot to show all relationships
print("\nPairplot showing all feature relationships:")
sns.pairplot(iris_df_clean, hue='species')
plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
plt.show()

## Summary of Findings
print("\nSummary of Findings:")
print("1. The three iris species are clearly distinguishable by their measurements, especially petal dimensions.")
print("2. Setosa has the smallest flowers with short, wide petals compared to the other species.")
print("3. Virginica has the largest flowers on average.")
print("4. There's a strong positive correlation between petal length and petal width across all species.")
print("5. The distributions show some overlap between versicolor and virginica, but setosa is quite distinct.")