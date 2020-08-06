from catboost import CatBoostClassifier
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np


class Model:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.model.load_model(os.getcwd() + '\\model.bkp')
        self.old_data = pd.read_excel(os.getcwd() + '\\old_data.xlsx')

    def predict(self, data):
        return self.model.predict(data)

    def retrain(self, data):
        new_data = pd.concat([self.old_data, data], axis=0, ignore_index=True)
        self.old_data = new_data.drop(
            labels=['ID (Идентификатор Заявки)', 'ID (Идентификатор Клиента)', 'Дата заявки', 'Unnamed: 0', '  - count', '  - summ'],
            axis='columns')
        X = self.old_data.drop(labels='Target (90 mob 12)', axis='columns')
        y = self.old_data['Target (90 mob 12)']
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7)
        X.to_excel(os.getcwd() + '\\mmmm.xlsx')
        categorical_features_indices = []
        print(categorical_features_indices, X)
        self.model = CatBoostClassifier(
           thread_count=2,
           iterations=50,
           depth=1,
           l2_leaf_reg = 2,
           learning_rate = 0.001,
           random_seed=62,
           od_type='Iter',
           od_wait=10,
           custom_loss=[ 'F1', 'AUC'],
           auto_class_weights='Balanced',
           use_best_model=True,
        )

        self.model.fit(
            X_train, y_train,
            cat_features=categorical_features_indices,
            eval_set=(X_val, y_val),
            logging_level='Silent',
            plot=True
        )
        self.model.save_model(os.getcwd() + '\\model.bkp')
        self.old_data.to_excel(os.getcwd() + '\\old_data.xlsx', index_label=False)


class Data:
    def __init__(self, data, transactions_data):
        self.data = data
        self.data['Сумма транзакций до выдачи кредита'] = 0
        self.data['Средняя стоимость транзакции до выдачи кредита'] = 0
        self.data['Минимальная стоимость транзакции до выдачи кредита'] = 0
        self.data['Максимальная стоимость транзакции до выдачи кредита'] = 0
        self.data['Количество транзакций до выдачи кредита'] = 0

        transactions_data['Тип операции'] = transactions_data['Тип операции'].fillna(' ')
        operation_types = transactions_data['Тип операции'].unique()
        transactions_data['Валюта платежной системы'] = transactions_data['Валюта платежной системы'].fillna(' ')
        money_types = transactions_data['Валюта платежной системы'].unique()
        transactions_data['Бизнес-тип операции'] = transactions_data['Бизнес-тип операции'].fillna(' ')
        business_types = transactions_data['Бизнес-тип операции'].unique()
        transactions_data['Тип карты'] = transactions_data['Тип карты'].fillna(' ')
        card_types = transactions_data['Тип карты'].unique()

        for type_ in operation_types:
            self.data[type_ + ' - count'] = 0
        for money_type in money_types:
            self.data[money_type + ' - summ'] = 0
        for business_type in business_types:
            self.data[business_type + ' - summ'] = 0
        for card_type in card_types:
            self.data[card_type + ' - summ'] = 0

        self.data['Кредитовая карта - summ'] = 0
        self.data['Овердрафт - summ'] = 0

        self.data = self.data.drop(labels=['SVC_CLSS_ID - count', 'SETL_CCY_ID - summ', 'Card_Type - summ'], axis='columns')
        self.data = self.data.drop(labels=['TXN_OPRN_TP_ID - summ'], axis='columns')

        transactions_data['Дата транзакции'] = transactions_data['Дата транзакции'].astype(str)
        i = 0
        for row in self.data.iterrows():
            row = row[1]

            id = row['ID (Идентификатор Заявки)']
            client_id = row['ID (Идентификатор Клиента)']
            date = row['Дата заявки']

            transactions = transactions_data[(transactions_data['Идентификатор Клиента'] == client_id) & (
                    transactions_data['Дата транзакции'] < str(date))]
            #print(transactions)
            # =======================================================Part 1
            self.data.loc[i, 'Сумма транзакций до выдачи кредита'] = transactions['Сумма транзакции'].sum()
            self.data.loc[i, 'Минимальная стоимость транзакции до выдачи кредита'] = transactions['Сумма транзакции'].min()
            self.data.loc[i, 'Максимальная стоимость транзакции до выдачи кредита'] = transactions['Сумма транзакции'].max()
            self.data.loc[i, 'Средняя стоимость транзакции до выдачи кредита'] = transactions['Сумма транзакции'].mean()
            self.data.loc[i, 'Количество транзакций до выдачи кредита'] = len(transactions['Сумма транзакции'])
            # =============================================================================================================================================

            # =======================================================Part 2
            self.data.loc[i, 'Transaction - count'] = len(transactions[transactions['Тип операции'] == 'Transaction'])
            self.data.loc[i, 'Misc - count'] = len(transactions[transactions['Тип операции'] == 'Misc'])

            self.data.loc[i, 'Russian Ruble - summ'] = \
                transactions[transactions['Валюта платежной системы'] == 'Russian Ruble'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Не определено в системе - summ'] = \
                transactions[transactions['Валюта платежной системы'] == 'Не определено в системе'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'US Dollar - summ'] = transactions[transactions['Валюта платежной системы'] == 'US Dollar'][
                'Сумма транзакции'].sum()
            self.data.loc[i, 'EURO - summ'] = transactions[transactions['Валюта платежной системы'] == 'EURO'][
                'Сумма транзакции'].sum()

            # =============================================================================================================================================

            # ========================================================Part 3
            self.data.loc[i, 'Оплата через Госуслуги - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Оплата через Госуслуги']['Сумма транзакции'].sum()
            self.data.loc[i, 'Покупка в ТСП через чужой POS - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Покупка в ТСП через чужой POS'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Зачисление субсидии на счетовой контракт - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Зачисление субсидии на счетовой контракт'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Прочее - summ'] = transactions[transactions['Бизнес-тип операции'] == 'Прочее'][
                'Сумма транзакции'].sum()
            self.data.loc[i, 'Перевод от клиента АББ НЕ клиенту АББ через АБО - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Перевод от клиента АББ НЕ клиенту АББ через АБО'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Оплата через Госуслуги РТ - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Оплата через Госуслуги РТ'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Зачисление средств на карточный счет через АТМ АББ - summ'] = \
                transactions[
                    transactions['Бизнес-тип операции'] == 'Зачисление средств на карточный счет через АТМ АББ'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Зачисление средств на карточный счет через инфокиоск АББ - summ'] = \
                transactions[
                    transactions['Бизнес-тип операции'] == 'Зачисление средств на карточный счет через инфокиоск АББ'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Покупка в ТСП через POS АББ - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Покупка в ТСП через POS АББ'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Снятие средств с карточного счета через АТМ АББ - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Снятие средств с карточного счета через АТМ АББ'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Зачисление ЗПК на счетовой контракт - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Зачисление ЗПК на счетовой контракт'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Перевод от клиента АББ клиенту АББ через АБО - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Перевод от клиента АББ клиенту АББ через АБО'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Перевод на свою карту через АБО - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Перевод на свою карту через АБО'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Снятие средств с карточного счета через чужой АТМ - summ'] = \
                transactions[
                    transactions['Бизнес-тип операции'] == 'Снятие средств с карточного счета через чужой АТМ'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Перевод от клиента АББ клиенту АББ через АТМ - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Перевод от клиента АББ клиенту АББ через АТМ'][
                    'Сумма транзакции'].sum()
            self.data.loc[i, 'Перевод на свою карту через АТМ АББ - summ'] = \
                transactions[transactions['Бизнес-тип операции'] == 'Перевод на свою карту через АТМ АББ'][
                    'Сумма транзакции'].sum()
            # ===============================================================================================================================================

            # ==========================================================Part 4
            self.data.loc[i, 'Дебетовая карта - summ'] = transactions[transactions['Тип карты'] == 'Дебетовая карта'][
                'Сумма транзакции'].sum()
            self.data.loc[i, 'Кредитовая карта - summ'] = transactions[transactions['Тип карты'] == 'Кредитовая карта'][
                'Сумма транзакции'].sum()
            self.data.loc[i, 'Овердрафт - summ'] = transactions[transactions['Тип карты'] == 'Овердрафт'][
                'Сумма транзакции'].sum()
            self.data.loc[i, 'Не определено в системе - summ'] = \
                transactions[transactions['Тип карты'] == 'Не определено в системе']['Сумма транзакции'].sum()

            # ===============================================================================================================================================

            i += 1

        self.data['К-во кредитов закрытых ранее'] = 0
        self.data['К-во кредитов не закрытых ранее'] = 0

        i = 0
        for row in self.data.iterrows():
            row = row[1]
            id = row['ID (Идентификатор Заявки)']
            client_id = row['ID (Идентификатор Клиента)']
            date = row['Дата заявки']

            credits_list = data[(self.data['ID (Идентификатор Клиента)'] == client_id) & (self.data['Дата заявки'] < str(date))]
            self.data.loc[i, 'К-во кредитов не закрытых ранее'] = credits_list['Target (90 mob 12)'].sum()
            self.data.loc[i, 'К-во кредитов закрытых ранее'] = \
                        credits_list.shape[0] - credits_list['Target (90 mob 12)'].sum()

            i += 1

    def get_data(self):
        return self.data


model = Model()
xls = pd.ExcelFile('2 (test).xlsx')
data = pd.read_excel(xls, 'test')
transactions_data = pd.read_excel(xls, 'транзакции (test)')

data = Data(data, transactions_data)
data = data.get_data()
data.to_excel('temp.xlsx')
data = data.drop(labels=['ID (Идентификатор Заявки)', 'ID (Идентификатор Клиента)', 'Дата заявки'], axis='columns')
y = model.predict(data)
data['Target (90 mob 12)'] = 0
for i in range(len(y)):
    data.loc[i, 'Target (90 mob 12)'] = y[i]



for x in y:
    print(x)
data.to_excel(os.getcwd() + '\\result.xlsx', index_label=False)
