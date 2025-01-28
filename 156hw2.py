import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#a
wine_data = pd.read_csv("winequality-red.csv", sep=";")
print("\na) ")
print(wine_data.head())

#b
from sklearn.model_selection import train_test_split
X = wine_data.drop('quality', axis = 1)
Y = wine_data['quality']

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size= 0.4, random_state= 42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size= 0.5, random_state = 42)

print("\nb) ")
print("Training Set Shape: ", X_train.shape, Y_train.shape)
print("Validation Set Shape: ", X_val.shape, Y_val.shape)
print("Test Set Shape: ", X_test.shape, Y_test.shape)

#c
X_train_np = X_train.to_numpy()
Y_train_np = Y_train.to_numpy()

X_train_bias = np.c_[np.ones(X_train_np.shape[0]), X_train_np]

# want to have w = (X^T * X)^(-1) * X^T * y
# first X^T * X
XtX = np.dot(X_train_bias.T, X_train_bias)

# next compute the inverse of X^T * X
XtX_inv = np.linalg.inv(XtX)

# then X^T * y
XtY = np.dot(X_train_bias.T, Y_train_np)

# find coefficients (w weights in formula)
Weights = np.dot(XtX_inv, XtY)
print("\nc) ")
print("Weights (coefficients):")
print(Weights)

inter = Weights[0]
coef = Weights[1:]
print("\nIntercept:", inter)
print("Coefficients: ", coef)

# function def for closed-form solu
def predict(X, Weights):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return np.dot(X_bias, Weights)

# apply function
Y_predict = predict(X_val.to_numpy(), Weights)

# find mean squared error 
MSE = np.mean((Y_val.to_numpy() - Y_predict) ** 2)
print("\n")
print("Mean Square Error: ", MSE)

#d
# went back and added this after creating and observing graph
print("\n")
print("d) Most of the values are clumped near the red line meaning that the model generally makes accurates predictions. The prediction errors are largest at 4 and 8 which is shown by higher deviation from the red line.")
# predicting target values of training data
Y_train_predict = predict(X_train_np, Weights)

# # plot values against each other
# plt.figure(figsize=(6, 4))
# plt.scatter(Y_train_np, Y_train_predict, alpha = 0.8, color= "blue")
# plt.plot([min(Y_train_np), max(Y_train_np)], [min(Y_train_np), max(Y_train_np)], color= "red", linestyle = "--")
# plt.title("Actual vs. Predicted Target Values") #(Training Data)
# plt.xlabel("Actual Target Values") #(Y_train)
# plt.ylabel("Predicted Target Values") #(Y_train_predict)
# plt.grid(True)
# plt.show()

#e
# function to calculate root mean square
def calculate_rms(Y_actual, Y_pred):
    MSE = np.mean((Y_actual - Y_pred) ** 2) # mean squared error
    RMS = np.sqrt(MSE) # root of mean squared error
    return RMS

# predict on train set
Y_train_predict = predict(X_train_np, Weights)
RMS_train = calculate_rms(Y_train_np, Y_train_predict)

# predict on test set
Y_test_predict = predict(X_test.to_numpy(), Weights)
RMS_test = calculate_rms(Y_test.to_numpy(), Y_test_predict)

print("\ne)")
print(f"Root Mean Square Error on Training Set: {RMS_train}")
print(f"Root Mean Square Error on Test Set: {RMS_test}")

# f
from sklearn.preprocessing import StandardScaler

# function to normalize data
def  normalize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# function for least-mean-squares algorithm implementation
def least_mean_squares(X, Y, learning_rate= 0.01, epochs = 1000):
    """""
    Function implements Least-Mean-Squares algorithm for linear regression.
    Parameters:
    1. X: Feature matrix (numpy array) with shape (n_samples, n_features)
    2. Y: Target vector (numpy array) with shape (n_samples)
    3. learning_rate: Step size for gradient descent
    4. Epochs: number of iterations

    Returns: 
    1. Weights: learned weights (numpy array)
    2. Errors: List of Mean Square Errors for each epoch (iteration)
    """

    # random initialization of weights
    np.random.seed(42)
    Weights = np.random.randn(X.shape[1] + 1) 

    # include bias to X
    X_bias = np.c_[np.ones(X.shape[0]), X]

    errors = [] #list initialization for storing MSE's
    for epoch in range(epochs):
        Y_predict = np.dot(X_bias, Weights)

        error = Y - Y_predict #residuals

        # update the weights
        gradient = -2 * np.dot(X_bias.T, error) / X.shape[0]
        Weights -= learning_rate * gradient

        # store mean squared error
        MSE = np.mean(error**2)
        errors.append(MSE)

    return Weights, errors

X_train_np_scaled = normalize_features(X_train_np) #normalizing features

learning_rate = 0.001 # tested out different ones 
epochs = 1000
Weights_LMS, MSE_errors = least_mean_squares(X_train_np_scaled, Y_train_np, learning_rate, epochs)

print("\nf) ")
print("Learned Weights:", Weights_LMS)

# # plot the mean squared estimate vs epochs
# plt.figure(figsize=(6,4))
# plt.plot(range(len(MSE_errors)), MSE_errors, color = "blue")
# plt.title("MSE vs. Epochs for LMS Algorithm")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Square Error")
# plt.grid(True)
# plt.show()

#g

# function to calculate root mean square
def calculate_rms(Y_actual, Y_predict):
    MSE = np.mean((Y_actual - Y_predict) ** 2) 
    RMS = np.sqrt(MSE) 
    return RMS

# predict on train and test set
def predict_LMS(X, Weights):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return np.dot(X_bias, Weights)

# normalize test features with scalar used in training
X_test_np_scaled = normalize_features(X_test.to_numpy())

# predict on train set
Y_train_predict = predict_LMS(X_train_np_scaled, Weights_LMS) # prediction for training set
RMS_train = calculate_rms(Y_train_np, Y_train_predict)

# prediction on test set
Y_test_predict = predict_LMS(X_test_np_scaled, Weights_LMS) 
RMS_test = calculate_rms(Y_test.to_numpy(), Y_test_predict)

print("\ng) ")
print(f"Root Mean Square Error on Training Set: {RMS_train}")
print(f"Root Mean Square Error on Test Set: {RMS_test}")

# graph for d
# plot values against each other
plt.figure(figsize=(6, 4))
plt.scatter(Y_train_np, Y_train_predict, alpha = 0.8, color= "blue")
plt.plot([min(Y_train_np), max(Y_train_np)], [min(Y_train_np), max(Y_train_np)], color= "red", linestyle = "--")
plt.title("Actual vs. Predicted Target Values") #(Training Data)
plt.xlabel("Actual Target Values") #(Y_train)
plt.ylabel("Predicted Target Values") #(Y_train_predict)
plt.grid(True)
plt.show()


# graph for f
# plot the mean squared estimate vs epochs
plt.figure(figsize=(6,4))
plt.plot(range(len(MSE_errors)), MSE_errors, color = "blue")
plt.title("MSE vs. Epochs for LMS Algorithm")
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error")
plt.grid(True)
plt.show()