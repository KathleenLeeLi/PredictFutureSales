#!/usr/bin/env python3


import pandas as pd

train = pd.read_csv('../data/sales_train.csv')
test = pd.read_csv('../data/test.csv')
submission = pd.read_csv('../data/sample_submission.csv')
items = pd.read_csv('../data/items.csv')
item_cats = pd.read_csv('../data/item_categories.csv')
shops = pd.read_csv('../data/shops.csv')