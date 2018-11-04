import numpy as np

class NormalLinearRegression():
    """
    Would you believe This is My Linear Regression
    """
    
    def __init__(self,add_bias = True):
        """
        This is an __init__ function it is same as constructor in Java
        """
        self.add_bias = add_bias
    def fit(self,X,y):
        """
        This is the function to fit the model parameters
        """
        if self.add_bias:
            # Add the bias term
            X = np.c_[np.ones((X.shape[0],X.shape[1])),X]
        self.theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
    def predict(self,X):
        """
        This function predict the value of Y
        """
        if not isinstance(X,np.ndarray):
            X = np.asarray(X)
        
        if self.add_bias:
            X = np.c_[np.ones((X.shape[0],X.shape[1])),X]
        y_predict = X.dot(self.theta_best)
        return y_predict
        
        
