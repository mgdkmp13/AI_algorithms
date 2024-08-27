import numpy as np

def calculate_MSE(theta, X, y):
    predicted_output = X @ theta
    MSE = np.mean((predicted_output-y) ** 2)
    return MSE

def calculate_gradient_BGD(theta_gr, learning_rate, X, y):
    last_cost = 0
    while True:
        gradient = (2 / len(y)) * X.T @ (X @ theta_gr - y)
        theta_gr = theta_gr - learning_rate * gradient

        cost = calculate_MSE(theta_gr, X, y)

        if abs(last_cost - cost) <= 0.0000001:
            break

        last_cost = cost
    return theta_gr
