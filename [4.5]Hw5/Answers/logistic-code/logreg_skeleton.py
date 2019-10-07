import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    n = len(y)
    total = 0
    for i in range(n):
        s = y[i] * theta.T @ X[i]
        x_star = (max(-700, - s - 700) + min(30, 30 - s)) / 2
        total += x_star + np.logaddexp(-x_star, -s - x_star)

    total *= 1 / n
    total += l2_param * theta.T @ theta
    return total


def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter

    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    def logistic_obj(w):
        return objective_function(w, X, y, l2_param)
    w_0 = np.zeros(len(X[0]))
    return minimize(logistic_obj, w_0).x


def get_data():
    X_train = np.loadtxt('X_train.txt', delimiter=',')
    X_val = np.loadtxt('X_val.txt', delimiter=',')
    y_train = np.loadtxt('y_train.txt', delimiter=',')
    y_val = np.loadtxt('y_val.txt', delimiter=',')
    # 0 -> -1
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1

    for i in range(len(y_val)):
        if y_val[i] == 0:
            y_val[i] = -1
    # standardizing
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train_sd = (X_train - means) / stds
    X_val_sd = (X_val - means) / stds
    # add 1 col
    X_train_sd = np.hstack((np.ones(len(X_train)).reshape(-1, 1), X_train_sd))
    X_val_sd = np.hstack((np.ones(len(X_val)).reshape(-1, 1), X_val_sd))

    return X_train_sd, X_val_sd, y_train, y_val


def get_performance(X_train_sd, X_val_sd, y_train, y_val, l2_param=1):
    w = fit_logistic_reg(X_train_sd, y_train, f_objective, l2_param=l2_param)
    # training error

    def cut(x):
        if x > 0.5:
            return 1
        else:
            return -1
    train_probs = 1 / (1 + np.exp(-1 * X_train_sd @ w))
    train_preds = np.array(list(map(cut, train_probs)))
    train_error = sum(train_preds != y_train) / len(X_train_sd)
    # val error
    val_probs = 1 / (1 + np.exp(-1 * X_val_sd @ w))
    val_preds = np.array(list(map(cut, val_probs)))
    val_error = sum(val_preds != y_val) / len(X_val_sd)
    print(f"training error: {train_error}\nval_error: {val_error}")
    return train_probs, val_probs


# best_l2 = 0.017
def find_best_l2(start=0.001, stop=1, n=10):
    l2_params = np.linspace(start, stop, n)
    X_train_sd, X_val_sd, y_train, y_val = get_data()
    losts = []
    for l2 in l2_params:
        w = fit_logistic_reg(X_train_sd, y_train, f_objective, l2_param=l2)
        lost = (f_objective(w, X_val_sd, y_val, l2) -
                l2 * w.T @ w) * len(y_val)
        losts.append(lost)
    plt.plot(l2_params, losts, 'o-')
    plt.show()


def calibration():
    X_train_sd, X_val_sd, y_train, y_val = get_data()
    w = fit_logistic_reg(X_train_sd, y_train, f_objective, l2_param=0.017)
    val_probs = 1 / (1 + np.exp(-1 * X_val_sd @ w))
    bins = np.arange(0.05, 1.1, 0.1)
    inds = np.digitize(val_probs, bins)
    pred_prob = [[0, 0] for _ in range(10)]
    for i, ind in enumerate(inds):
        pred_prob[ind - 1][0] += val_probs[i]
        pred_prob[ind - 1][1] += 1

    mean_probs = [p[0] / p[1] for p in pred_prob]
    expect_probs = np.arange(0.1, 1.1, 0.1)
    plt.plot(expect_probs, expect_probs,
             color='gray', marker='o', linestyle='dashed')
    plt.plot(expect_probs, mean_probs,
             color='green', marker='+')
    plt.show()


if __name__ == '__main__':
    X_train_sd, X_val_sd, y_train, y_val = get_data()
    p1, p2 = get_performance(X_train_sd, X_val_sd, y_train, y_val, 0.017)
    find_best_l2(0.01, 0.04, 5)
    calibration()
