import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1. Generate a 2-class synthetic Gaussian dataset
np.random.seed(42) # For reproducibility

# Class 0 Data
mean0 = [2, 2]
cov0 = [[1, 0.5], [0.5, 1]]
X0 = np.random.multivariate_normal(mean0, cov0, 150)
y0 = np.zeros(150)

# Class 1 Data
mean1 = [5, 4]
cov1 = [[1.5, -0.3], [-0.3, 1.5]]
X1 = np.random.multivariate_normal(mean1, cov1, 150)
y1 = np.ones(150)

# Combine the datasets
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# 2. Train the Bayes Classifier (Estimate Parameters)
# Calculate prior probabilities P(C_i)
prior0 = len(y0) / len(y)
prior1 = len(y1) / len(y)

# Calculate empirical means and covariance matrices
mu0 = np.mean(X[y == 0], axis=0)
mu1 = np.mean(X[y == 1], axis=0)
sigma0 = np.cov(X[y == 0], rowvar=False)
sigma1 = np.cov(X[y == 1], rowvar=False)

# Define the likelihood distributions p(x|C_i)
likelihood0 = multivariate_normal(mu0, sigma0)
likelihood1 = multivariate_normal(mu1, sigma1)

# 3. Create a meshgrid to classify test points and plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Flatten the grid to pass into the likelihood functions
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 4. Compute class probabilities (Posterior proportional to Likelihood * Prior)
posterior0 = likelihood0.pdf(grid_points) * prior0
posterior1 = likelihood1.pdf(grid_points) * prior1

# 5. Classify to minimize misclassification rate
# Assigns class 1 if posterior1 > posterior0, else class 0
predictions = (posterior1 > posterior0).astype(int)
Z = predictions.reshape(xx.shape)

# 6. Plotting the results
plt.figure(figsize=(10, 6))

# Plot the decision boundary and regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.contour(xx, yy, Z, colors='k', linewidths=1) # The exact boundary line

# Plot the original dataset points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', edgecolor='k', alpha=0.7)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', edgecolor='k', alpha=0.7)

plt.title('Bayes Classifier Decision Boundary\n(Synthetic Gaussian Dataset)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()