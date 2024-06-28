# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 載入 iris 資料集
iris = load_iris()
X = iris.data
y = iris.target

# 使用 PCA 降維成 2 維
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=100)

weights_list = ['uniform', 'distance']
algorithm_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
metric_list = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

for weights in weights_list:
    for algorithm in algorithm_list:
        for metric in metric_list:
            # 建立 KNN 模型，并?置??
            model = KNeighborsClassifier(n_neighbors=5, weights=weights, metric=metric, algorithm=algorithm)

            # 訓練模型
            model.fit(X_train, y_train)

            # 預測測試集
            y_pred = model.predict(X_test)

            # 計算準確率
            accuracy = accuracy_score(y_test, y_pred)
            print('Parameters: weights={}, algorithm={}, metric={}'.format(weights, algorithm, metric))
            print('Accuracy:', accuracy)
            print()