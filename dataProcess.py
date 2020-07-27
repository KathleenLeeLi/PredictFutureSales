#!/usr/bin/env python3

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--init',type=float,default=0.1,help='initial weight size')
parser.add_argument('--hid',type=int,default='10',help='number of hidden units')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
parser.add_argument('--epochs',type=int,default='100000',help='max training epochs')
args = parser.parse_args()

train = pd.read_csv('../data/sales_train.csv')
test = pd.read_csv('../data/test.csv')
submission = pd.read_csv('../data/sample_submission.csv')
items = pd.read_csv('../data/items.csv')
item_cats = pd.read_csv('../data/item_categories.csv')
shops = pd.read_csv('../data/shops.csv')