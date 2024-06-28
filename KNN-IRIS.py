# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ���J iris ��ƶ�
iris = load_iris()
X = iris.data
y = iris.target

# �ϥ� PCA ������ 2 ��
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# �����V�m���P���ն�
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=100)

weights_list = ['uniform', 'distance']
algorithm_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
metric_list = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

for weights in weights_list:
    for algorithm in algorithm_list:
        for metric in metric_list:
            # �إ� KNN �ҫ��A�}?�m??
            model = KNeighborsClassifier(n_neighbors=5, weights=weights, metric=metric, algorithm=algorithm)

            # �V�m�ҫ�
            model.fit(X_train, y_train)

            # �w�����ն�
            y_pred = model.predict(X_test)

            # �p��ǽT�v
            accuracy = accuracy_score(y_test, y_pred)
            print('Parameters: weights={}, algorithm={}, metric={}'.format(weights, algorithm, metric))
            print('Accuracy:', accuracy)
            print()