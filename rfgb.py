# random forest gradient boost

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def param_plot(lab):
    plt.legend(lab, loc='upper right')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.show()


def param_alg_eval(param_grid, n_estimators, x_train, y_train, x_test, y_test, flag_grad=False):
    for param in param_grid:
        value = param_grid[param]
        labels = []
        for val in value:
            accuracy_train, accuracy_test = [], []
            for estimator in n_estimators:
                params = {param: val, 'n_estimators': estimator}
                if flag_grad:
                    rfc_model = GradientBoostingClassifier(**params)
                else:
                    rfc_model = RandomForestClassifier(**params)
                rfc_model.fit(x_train, y_train)
                accuracy_train.append(accuracy_score(y_train, rfc_model.predict(x_train)))
                accuracy_test.append(accuracy_score(y_test, rfc_model.predict(x_test)))
            plt.figure('train')
            plt.plot(n_estimators, accuracy_train)
            plt.figure('test')
            plt.plot(n_estimators, accuracy_test)
            labels.append(param + ' ' + str(round(val, 2)) if isinstance(val, int) or isinstance(val, float) else val)
        param_plot(labels)


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

    n_estimators = [int(x) for x in np.linspace(start=1, stop=200, num=50)]

    rf_accuracy_train, rf_accuracy_test, labels = [], [], []
    gb_accuracy_train, gb_accuracy_test = [], []
    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator)
        gb = GradientBoostingClassifier(n_estimators=estimator)
        rf.fit(x_train, y_train)
        gb.fit(x_train, y_train)
        rf_accuracy_train.append(accuracy_score(y_train, rf.predict(x_train)))
        gb_accuracy_train.append(accuracy_score(y_train, gb.predict(x_train)))
        rf_accuracy_test.append(accuracy_score(y_test, rf.predict(x_test)))
        gb_accuracy_test.append(accuracy_score(y_test, gb.predict(x_test)))
    plt.figure('rf train')
    plt.plot(n_estimators, rf_accuracy_train)
    plt.figure('rf test')
    plt.plot(n_estimators, rf_accuracy_test)
    labels.append('n_estimators' + ' ' + str(round(estimator, 2)))
    param_plot(labels)
    plt.figure('gb train')
    plt.plot(n_estimators, gb_accuracy_train)
    plt.figure('gb test')
    plt.plot(n_estimators, gb_accuracy_test)
    labels.append('n_estimators' + ' ' + str(round(estimator, 2)))
    param_plot(labels)

    n_train_max = n_estimators[rf_accuracy_train.index(max(rf_accuracy_train))]
    n_test_max = n_estimators[rf_accuracy_test.index(max(rf_accuracy_test))]

    print('rf', n_train_max, n_test_max)

    print('gb', n_estimators[gb_accuracy_train.index(max(gb_accuracy_train))],
          n_estimators[gb_accuracy_test.index(max(gb_accuracy_test))])

    print(max(n_train_max, n_test_max) + 20)

    n_estimators = [int(x) for x in np.linspace(start=1, stop=max(n_train_max, n_test_max) + 20, num=50)]

    random_forest_grid = {'max_depth': [int(x) for x in np.linspace(start=1, stop=20, num=4)],
                          'max_features': [1, 'sqrt', 6]}

    param_alg_eval(random_forest_grid, n_estimators, x_train, y_train, x_test, y_test)

    gradient_boost_grid = {'max_depth': [int(x) for x in np.linspace(start=1, stop=20, num=4)],
                           'learning_rate': [float(x) for x in np.linspace(start=0.01, stop=0.5, num=4)]}

    param_alg_eval(gradient_boost_grid, n_estimators, x_train, y_train, x_test, y_test, True)


if __name__ == "__main__":
    main()
