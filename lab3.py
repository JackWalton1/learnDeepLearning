import numpy as np

# Classic gradient descent 
def gradient_descent(X, y, learning_rate, num_iterations):
    """ Function from LinkedIn article (The classic weight optimizer algo for DL): 
    https://www.linkedin.com/pulse/gradient-descent-its-applications-deep-learning-chirag-subramanian/
    """
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)
    for _ in range(num_iterations):
        gradient = np.dot(X.T, (np.dot(X, theta) - y)) / num_samples
        theta -= learning_rate * gradient

    return theta

# Batch GD
def batch_gradient_descent(X, y, learning_rate, num_iterations):
    """ Variant: batch gradient descent.
        Makes more volatile updates to the weights.
    """
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)
    numRows = len(X)

    for i in range(num_iterations):
        learning_rate = learning_rate / (1+i / num_iterations)
        gradient = (1/numRows) * np.dot(X.T, (np.dot(X, theta) - y))
        theta -= learning_rate * gradient

    return theta

# SGD
def stochastic_gradient_descent(X, y, learning_rate, num_iterations):
    """ Variant: stochastic gradient descent (SGD) 
    It updates the parameters based on a randomly selected subset of training 
    samples in each iteration, rather than the entire dataset. This helps in 
    speeding up the training process and making it feasible for large-scale 
    problems.
    """
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)
    numRows = len(X)

    for _ in range(num_iterations):
        for i in range(numRows):
            gradient = X[i] * (np.dot(X[i], theta) - y[i])
            theta -= learning_rate * gradient

    return theta

# Mini-batch GD
def mini_batch_gradient_descent(X, y, learning_rate, num_iterations, batch_size = 20):
    """ Variant: mini-batch gradient descent.
    A balance by using a small batch of randomly selected samples for
    parameter updates. This approach combines the advantages of both batch
    gradient descent (accurate updates) and stochastic gradient descent 
    (faster convergence).
    """
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)
    numRows = len(X)

    for _ in range(num_iterations):
        indices = np.random.permutation(numRows)
        X = X[indices]
        y = y[indices]
        for i in range(0, numRows, batch_size):
            X_subset = X[i:i+batch_size]
            y_subset = y[i:i+batch_size]
            gradient = (1/len(X_subset)) * \
                            (np.dot(X_subset.T, np.dot(X_subset, theta) - y_subset))
            theta -= learning_rate * gradient

    return theta

"""attempting to make a random dataset"""
X = np.array([[0,2], [1,3], [1,5], [2, 6], [1, 9]]) # the avg of each row...
y = np.array([1, 2, 3, 4, 5]) # ...is this

"""Testing out n learning rates and m number of iterations"""
learning_rates = [.01, .02, .03, .04, .05, .07]
num_iterationsL = [1000, 2000, 3000, 5000, 10000, 20000]

"""setting these to null"""
optCost, optLearningRate, optNumIterations, optTheta = None, None, None, None

"""
Finding optimized params by running (n * m) combinations 
of learning rates and number of iterations.
"""
for learning_rate in learning_rates:
    for num_interations in num_iterationsL:
        """ 
        Since GD is the common underpinning of the variants, we will use
        it to find out what optimal params are
        """
        theta = gradient_descent(X, y, learning_rate, num_interations)
        numRows = len(X)
        cost = (1 / (2 * numRows)) * np.sum((np.dot(X, theta) - y) ** 2)
        if optCost is None or cost < optCost:
            optCost = cost
            optLearningRate = learning_rate
            optNumIterations = num_interations
            optTheta = theta

"""Printing optimized params of classic gradient descent"""
print("The optimal learning rate is: ", optLearningRate)
print("The optimal number of iterations is: ", optNumIterations)
print("Optimized (classic) gradient descent yeilds theta=", optTheta)

"""Computing theta of gradient descent variants"""
thetaBGD = batch_gradient_descent(X, y, optLearningRate, optNumIterations)
thetaSGD = stochastic_gradient_descent(X, y, optLearningRate, optNumIterations)
thetaMBGD = mini_batch_gradient_descent(X, y, optLearningRate, optNumIterations)

"""Printing theta of gradient descent variants"""
print("Optimized BGD yeilds theta=", thetaBGD)
print("Optimized SGD yeilds theta=", thetaSGD)
print("Optimized MBGD yeilds theta=", thetaMBGD)