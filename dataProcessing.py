#!/usr/bin/env python3

import pandas as pd
import numpy as np

def loadData():
    train = pd.read_csv('./data/sales_train.csv')
    test = pd.read_csv('./data/test.csv')
    submission = pd.read_csv('./data/sample_submission.csv')
    items = pd.read_csv('./data/items.csv')
    itemCategory = pd.read_csv('./data/item_categories.csv')
    shops = pd.read_csv('./data/shops.csv')
    return train, test, submission, items, itemCategory, shops

# Data exploration
def dataExploration(df):
    print('Samle data')
    print(df.head())
    print('\nDescribe data')
    print(df.describe().T)
    print('\nNull data in set:')
    print(df.isnull().sum())

# Data preprocessing
def dataProcess(train, test):
    # reset data type of 'date to date frame
    train['date'] = pd.to_datetime(train['date'],format = '%d.%m.%Y')

    # merge rows with same shop_id and item_id
    dataset = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')
    dataset.reset_index(inplace=True)
    print(dataset)

    # join with test table to see matching for store and item
    dataset = pd.merge(test, dataset, on=['shop_id', 'item_id'], how='left')
    dataset.fillna(0, inplace=True)
    print(dataset)

    # drop id as well because ID = index
    dataset.drop(['shop_id','item_id','ID'], inplace=True, axis=1)
    print(dataset)

    return dataset

# def dataScaling(data):
#     scaler = MinMaxScaler(feature_range=(-1,1))
#     normalisedData = scaler.fit_transform(data.reshape(-1,1))

#     return normalisedData

# def createTimeSeries(data, timestep):
#     X=[]
#     Y=[]
#     for i in range(len(normalised_X) - timestep):
#         X.append(data[i:i+timestep])
#         Y.append(data[i+timestep])
    
#     X=np.asanyarray(X)
#     Y=np.asanyarray(Y)

#     return X,Y