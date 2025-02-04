import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def compute_loss(Y, Y_prediction):
    return -np.mean(Y * np.log(Y_prediction + 1e-9) + (1 - Y) * np.log(1 - Y_prediction + 1e-9))

def logistic_regression(X, Y, batch_size, fixed_learning_rate, max_iterations):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features)
    w0 = np.random.randn()
    
    for _ in range(max_iterations):
        indices = np.random.permutation(n_samples)
        X_mixed, Y_mixed = X[indices], Y[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_mixed[i:i + batch_size]
            Y_batch = Y_mixed[i:i + batch_size]
            
            output = np.dot(X_batch, w) + w0
            Y_prediction = sigmoid(output)
            
            # Compute gradients
            gradient_w = np.dot(X_batch.T, (Y_prediction.flatten() - Y_batch.flatten())) / batch_size #needed to flatten to match broadcasting shapes
            gradient_w0 = np.mean(Y_prediction.flatten() - Y_batch.flatten())
            # Update parameters
            w = w - fixed_learning_rate * gradient_w
            w0 = w0 -  fixed_learning_rate * gradient_w0
    
    return w, w0

#a) load dataset
data = load_breast_cancer()
X = data.data  
Y = data.target  

#b) splitting into train and test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

#c) report class size
print("Class distribution of Training and Validation sets:")
print("Class 0:", np.sum(Y_train == 0))
print("Class 1:", np.sum(Y_train == 1))

#d) training a binary logistic regression model from q3 
# initialize the model weights randomly, sample from std. gaussian distribution
w, w0 = logistic_regression(X_train, Y_train, batch_size=10, fixed_learning_rate=0.1, max_iterations=1000)

#e) evaluate on test set
Y_test_pred = sigmoid(np.dot(X_test, w) + w0) > 0.5

accuracy = accuracy_score(Y_test, Y_test_pred)
precision = precision_score(Y_test, Y_test_pred)
recall = recall_score(Y_test, Y_test_pred)
f1 = f1_score(Y_test, Y_test_pred)

# print out
print(f"Accuracy: {accuracy:}")
print(f"Precision: {precision:}")
print(f"Recall: {recall:}")
print(f"F1-score: {f1:}")
