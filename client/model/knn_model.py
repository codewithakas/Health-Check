import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
import joblib
# Load dataset
data = pd.read_csv("diabetes.csv", sep=",", header=0)

# Remove the SkinThickness column
data = data.drop("SkinThickness", axis=1)

# Split dataset into features and target variable
X = data.iloc[:, :-1].values  # features
y = data.iloc[:, -1].values  # target variable

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0)


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = np.linalg.norm(self.X_train - sample, axis=1)
            nearest_neighbors = np.argsort(distances)[:self.k]
            neighbors_labels = self.y_train[nearest_neighbors]
            unique_labels, counts = np.unique(
                neighbors_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)
        return np.array(predictions)

# Finding the best K value
# best_accuracy = 0
# best_k = 1
# for k in range(1, 20):
#     knn = KNNClassifier(n_neighbors = k)
#     cv_scores = cross_val_score(knn, X_train, y_train, cv=10)
#     if np.mean(cv_scores) > best_accuracy:
#         best_accuracy = np.mean(cv_scores)
#         best_k = k


if __name__ == "__main__":

    # Training the KNN model on the training set with the best K value
    knn = KNNClassifier(k=6)
    knn.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = knn.predict(X_test)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Best K value: {best_k}')
    print(f'Test Set Accuracy: {accuracy * 100}%')

    # Save the KNN model
    joblib.dump([knn, accuracy, scaler], 'knn_model.pkl')
