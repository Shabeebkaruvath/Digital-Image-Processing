import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load Dataset (Breast Cancer)
# Class 0: Malignant (Severe), Class 1: Benign (Safe)
data = load_breast_cancer()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Perform Feature Selection
# Select the top 10 most significant features based on ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=10)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# 3. Apply LDA (Linear Discriminant Analysis)
# Reduces the feature space to maximize class separability
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_sel, y_train)
X_test_lda = lda.transform(X_test_sel)

# 4. Train a Bayes Classifier
nb = GaussianNB()
nb.fit(X_train_lda, y_train)

# Predict probabilities for the test set: [P(Class 0), P(Class 1)]
probas = nb.predict_proba(X_test_lda)

# Define the Loss Matrix
# Format: L[predicted_class, true_class]
# L[0, 0]: Predict Malignant, True Malignant (0 cost)
# L[0, 1]: Predict Malignant, True Benign (Cost=2, unnecessary panic/tests)
# L[1, 0]: Predict Benign, True Malignant (Cost=100, extremely dangerous false negative)
# L[1, 1]: Predict Benign, True Benign (0 cost)
loss_matrix = np.array([
    [0, 2],    # Expected losses for predicting Class 0
    [100, 0]   # Expected losses for predicting Class 1
])

# Compute Expected Risk for each class prediction
# expected_risks shape: (n_samples, n_classes)
expected_risks = np.dot(probas, loss_matrix.T)

# Classify points by picking the class that minimizes the expected risk
y_pred_loss_adjusted = np.argmin(expected_risks, axis=1)

# 5. Evaluate Performance
accuracy = accuracy_score(y_test, y_pred_loss_adjusted)
conf_matrix = confusion_matrix(y_test, y_pred_loss_adjusted)

# Compute the Expected Loss (Average penalty incurred on the test set)
actual_losses = [loss_matrix[pred, true] for pred, true in zip(y_pred_loss_adjusted, y_test)]
expected_loss = np.mean(actual_losses)

# Standard predictions (without loss matrix) for comparison
y_pred_standard = nb.predict(X_test_lda)
std_losses = [loss_matrix[pred, true] for pred, true in zip(y_pred_standard, y_test)]
std_expected_loss = np.mean(std_losses)

# Print Results
print("=== Classification Results with Loss Matrix ===")
print(f"Accuracy:           {accuracy * 100:.2f}%")
print(f"Expected Loss/Cost: {expected_loss:.4f} per sample")
print(f"Standard Bayes Expected Loss (No Matrix): {std_expected_loss:.4f} per sample")

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant (0)', 'Benign (1)'], 
            yticklabels=['Malignant (0)', 'Benign (1)'])
plt.title('Confusion Matrix (Loss-Adjusted)')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()