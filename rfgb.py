# random forest gradient boost

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy


def param_plot(lab):
    plt.legend(lab, loc='upper right')
    plt.xlabel('n_estimators')
    plt.ylabel('%')
    plt.show()


def main():
    df = pd.read_csv('data.csv', header=None)
    df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']
    boundary_index = round(df.shape[0] * 0.8)
    df = df.sample(frac=1)

    x = df.iloc[:, :-1].to_numpy()
    x = preprocessing.normalize(x)
    y = df['y'].to_numpy()
    x_train, y_train = x[:boundary_index], y[:boundary_index]
    x_test, y_test = x[boundary_index:], y[boundary_index:]

    # rfc_model = RandomForestClassifier()
    # hyper_params = rfc_model.get_params()
    #
    # print(hyper_params)

    # random forest

    param_grid = {'max_features': ['sqrt', None],
                  'max_depth': [int(x) for x in np.linspace(start=5, stop=100, num=2)],
                  'min_samples_split': [int(x) for x in np.linspace(start=2, stop=10, num=2)],
                  'min_samples_leaf': [int(x) for x in np.linspace(start=1, stop=10, num=2)],
                  'bootstrap': [True, False]}

    n_estimators = [int(x) for x in np.linspace(start=1, stop=150, num=50)]

    for param in param_grid:
        value = param_grid[param]
        labels = []
        for val in value:
            accuracy_train, accuracy_test = [], []
            for estimator in n_estimators:
                params = {param: val, 'n_estimators': estimator}
                rfc_model = RandomForestClassifier(**params)
                rfc_model.fit(x_train, y_train)
                accuracy_train.append(accuracy_score(y_train, rfc_model.predict(x_train)))
                accuracy_test.append(accuracy_score(y_test, rfc_model.predict(x_test)))
            plt.figure('train')
            plt.plot(n_estimators, accuracy_train)
            plt.figure('test')
            plt.plot(n_estimators, accuracy_test)
            labels.append(param + ' ' + str(val))
        param_plot(labels)



    # # 3960 kombinacija
    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=5)]
    # # Number of features to consider at every split
    # max_features = ['log2', 'sqrt', None]
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(start=50, stop=250, num=5)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=5)]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=10, num=5)]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    #
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #
    # rfc_random = RandomizedSearchCV(estimator=rfc_model, param_distributions=random_grid, n_iter=100, verbose=2,
    #                                 n_jobs=-1)
    # rfc_best_model = rfc_random.fit(x_train, y_train)
    # y_best_pred = rfc_best_model.predict(x_test)
    #
    # rfc_base_model = RandomForestClassifier()
    # rfc_base_model.fit(x_train, y_train)
    # y_base_pred = rfc_base_model.predict(x_test)
    #
    # print("FINISHED BASE classifying. accuracy score : ", accuracy_score(y_test, y_base_pred))
    # print("FINISHED OPTIMIZED classifying. accuracy score : ", accuracy_score(y_test, y_best_pred))

    #
    #
    # print(rfc_random.best_params_)

    # rfc_model.fit(x_train, y_train)
    #
    # y_pred = rfc_model.predict(x_test)
    #
    # print(rfc_model.get_params())
    #


if __name__ == "__main__":
    main()
