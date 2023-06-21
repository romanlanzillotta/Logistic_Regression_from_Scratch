import math
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = None
        self.epoch_error_first = None
        self.epoch_error_last = None
        self.accuracy = None
    def sigmoid(self, t):
        return 1/(1+math.exp(-t))

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        if self.fit_intercept:
            x_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
        else:
            x_train = X_train
        self.coef_ = np.zeros(x_train.shape[1])
        for epoch in range(self.n_epoch):
            for i, row in enumerate(x_train):
                # update all weights
                new_weights = self.coef_[:]
                y_est = self.predict_proba(row, new_weights)
                for j in range(len(new_weights)):
                    new_weights[j] = new_weights[j] - self.l_rate * (y_est - y_train[i]) * y_est * (1-y_est) * row[j]
                self.coef_ = new_weights
            if epoch in [0, (self.n_epoch-1)]:
                if epoch == 0:
                    self.epoch_error_first = self.calc_error_mse(self.coef_, x_train, y_train)
                else:
                    self.epoch_error_last = self.calc_error_mse(self.coef_, x_train, y_train)

    def calc_error_mse(self, coefs, x_train, y_train):
        error_mse = []
        sum_ = 0
        for i, row in enumerate(x_train):
            y_est = self.predict_proba(row, coefs)
            sum_ += (y_est - y_train[i])**2
            error_mse.append(sum_ / (i+1))
        return error_mse

    def fit_log_loss(self, X_train, y_train):
        if self.fit_intercept:
            x_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
        else:
            x_train = X_train
        self.coef_ = np.zeros(x_train.shape[1])
        N = x_train.shape[0]
        for epoch in range(self.n_epoch):
            for i, row in enumerate(x_train):
                # update all weights
                new_weights = self.coef_[:]
                y_est = self.predict_proba(row, new_weights)
                for j in range(len(new_weights)):
                    new_weights[j] = new_weights[j] - self.l_rate * (y_est - y_train[i]) * row[j] / N
                self.coef_ = new_weights
            if epoch in [0, (self.n_epoch-1)]:
                if epoch == 0:
                    self.epoch_error_first = self.calc_error_log_loss(self.coef_, x_train, y_train)
                else:
                    self.epoch_error_last = self.calc_error_log_loss(self.coef_, x_train, y_train)

    def calc_error_log_loss(self, coefs, x_train, y_train):
        error_log = []
        sum_ = 0
        for i, row in enumerate(x_train):
            y_est = self.predict_proba(row, coefs)
            sum_ += y_train[i] * np.log(y_est) + (1 - y_train[i]) * np.log(1 - y_est)
            error_log.append(- sum_ / (i+1))
        return error_log

    def predict(self, x_test, cut_off=0.5):
        predictions = np.array([])
        if self.fit_intercept:
            x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
        for row in x_test:
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat < cut_off:
                predictions = np.append(predictions, 0)
            else:
                predictions = np.append(predictions, 1)
        return predictions  # predictions are binary values - 0 or 1


def standarization(arr):
    return (arr-arr.mean())/arr.std()


# loading data and standarization
df = load_breast_cancer(return_X_y=True, as_frame=True)
columns = ["worst concave points", "worst perimeter", "worst radius"]
X = pd.DataFrame(df[0])
X = X.loc[:, columns].to_numpy()
for column_ in range(X.shape[1]):
    X[:, column_] = standarization(X[:, column_])
y = pd.DataFrame(df[1])
y = y.iloc[:, 0].to_numpy()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)


def fit_eval_model(estimator, fit_func, x_train, y_train, x_test, y_test):
    # fitting the model
    fit_func(x_train, y_train)
    # predictions and evaluation
    y_preds = estimator.predict(x_test)
    estimator.accuracy = accuracy_score(y_test, y_preds)
    # dict_ = {'coef_': list(estimator.coef_), 'accuracy': estimator.accuracy}
    # print(dict_)


# Custom logistic regression with MSE fit
dict_ = {}
lr_mse = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
fit_eval_model(lr_mse, lr_mse.fit_mse, X_train, y_train, X_test, y_test)
dict_['mse_accuracy'] = lr_mse.accuracy
# Custom logistic regression with log-loss fit
lr_logloss = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
fit_eval_model(lr_logloss, lr_logloss.fit_log_loss, X_train, y_train, X_test, y_test)
dict_['logloss_accuracy'] = lr_logloss.accuracy
# SKLEARN logistic regression
lr_sklearn = LogisticRegression(random_state=43, max_iter=1000)
fit_eval_model(lr_sklearn, lr_sklearn.fit, X_train, y_train, X_test, y_test)
dict_['sklearn_accuracy'] = lr_sklearn.accuracy

# Errors:
dict_['mse_error_first'] = lr_mse.epoch_error_first
dict_['mse_error_last'] = lr_mse.epoch_error_last
dict_['logloss_error_first'] = lr_logloss.epoch_error_first
dict_['logloss_error_last'] = lr_logloss.epoch_error_last

print(dict_)

# If your solution works properly, you will receive graph.jpg in the current directory.
# It shows the errors plotted on four graphs. Explore these plots and answer the following questions:
#
# What is the minimum MSE value for the first epoch?
# What is the minimum MSE value for the last epoch?
# What is the maximum Log-loss value for the first epoch?
# What is the maximum Log-loss value for the last epoch?
# Has the range of the MSE values expanded or narrowed? (expanded/narrowed)
# Has the range of the Log-loss values expanded or narrowed? (expanded/narrowed)

print("Answers to the questions:")
print("1) 0.0000")
print("2) 0.0000")
print("3) 0.001530")
print("4) 0.006")
print("5) expanded")
print("6) expanded")