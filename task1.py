from catboost import CatBoostClassifier
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np


class Model:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.model.load_model(os.getcwd() + '\\result.bkp')
        self.old_data = pd.read_excel(os.getcwd() + '\\1 (train).xlsx')

    def predict(self, data):
        return self.model.predict(data)

    def retrain(self, data):
        new_data = pd.concat([self.old_data, data], axis=0)
        self.old_data = new_data
        X = new_data.drop(labels=['Target', 'NaturalPersonID', 'RequestDate'], axis='columns')
        y = new_data['Target']
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7)

        categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
        print(categorical_features_indices, X.shape)
        self.model = CatBoostClassifier(
            thread_count=3,
            iterations=50,
            depth=2,
            l2_leaf_reg=2,
            learning_rate=0.001,
            random_seed=62,
            od_type='Iter',
            od_wait=1,
            custom_loss=['F1'],
            use_best_model=True,
            class_weights=[0.025, 1],
        )

        self.model.fit(
            X_train, y_train,
            cat_features=categorical_features_indices,
            eval_set=(X_val, y_val),
            logging_level='Silent',
            plot=True
        )
        self.model.save_model(os.getcwd() + '\\result.bkp')



class Data:
    def __init__(self, data):
        self.X = data.drop(labels=['Target', 'NaturalPersonID', 'RequestDate'], axis='columns')

    def get_data(self):
        return self.X


model = Model()

X = pd.read_excel(os.getcwd() + '\\1 (test).xlsx')
data = Data(X)
data = data.get_data()
y = model.predict(data)
for i in range(len(y)):
    data.loc[i, 'Target (90 mob 12)'] = y[i]
data.to_excel('result.xlsx')
print('Good work!')
