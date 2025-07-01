
K-Nearest Neighbors (KNN) Classification - Iris Dataset

Objective:
Implement and understand KNN classification, evaluate model performance, experiment with different K values, and visualize results.

Tools & Libraries:
- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib

Dataset:
Filename: Iris.csv
Description: Classic Iris flower dataset with 150 samples and 3 species.

Features Include:
- Sepal Length, Sepal Width, Petal Length, Petal Width
- Target Variable: Species (Setosa, Versicolor, Virginica)

Steps Performed:
✔️ Data Preprocessing and Encoding of Target
✔️ Feature Normalization using StandardScaler
✔️ Trained KNN Classifier with different K values
✔️ Evaluated Model using Accuracy and Confusion Matrix
✔️ Visualized Accuracy vs. K to select optimal K
✔️ Final Model tested with best K value
✔️ Confusion Matrix visualized for performance insight

Results Summary:
- Model tested for K values from 1 to 19
- Optimal K chosen based on testing accuracy curve
- Achieved high classification accuracy on Iris dataset
- Confusion Matrix confirms good model performance

How to Run:
1. Install required libraries:
   pip install pandas numpy scikit-learn matplotlib seaborn

2. Place 'Iris.csv' dataset in the same folder as the Python script.

3. Run the Python script:
   python Task6.py

Key Learnings:
- KNN is an instance-based, distance-driven algorithm
- Normalization crucial for distance-based models
- Proper K selection prevents underfitting or overfitting
- Visualization aids in interpreting model behavior

Submission Notes:
- Included clean Python code file
- Dataset used: Iris.csv
- Visualizations generated during runtime
- README provided explaining task workflow
