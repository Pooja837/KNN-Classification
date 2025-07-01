''' Required Libraries '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


''' Load Dataset '''

df = pd.read_csv('C:/Desktop/Elevate Labs/Iris.csv')


''' Drop ID Column '''

df.drop('Id', axis=1, inplace=True)


''' Encode Target if Needed '''

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])


''' Features and Target '''

X = df.drop('Species', axis=1)
y = df['Species']


''' Normalize Features '''

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


''' Train-Test Split '''

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


""" 1 KNN with Different K values """

train_accuracies = []
test_accuracies = []
k_values = range(1, 20)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))


''' Plot Accuracy vs. K '''

plt.figure(figsize=(10,6))
plt.plot(k_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(k_values, test_accuracies, label='Testing Accuracy', marker='s')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. K (Iris Dataset)')
plt.legend()
plt.grid()
plt.show()


""" 2 Final Model with Best K (Example: k=5) """

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("KNN Accuracy (k=5): {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


''' Confusion Matrix '''

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - KNN (k=5)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
