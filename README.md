# Predict Future Sales

COMP9417 Project Repository by Zixin Zhou (Cindy) and Sophia Tran

## Note
Due to restricted size of submission. The data we used for the task is uploaded in https://drive.google.com/drive/folders/1BMnVfSWcoCCJxN610-1hfqun9NVS3Bg9?usp=sharing 

Please download from link, unzip and store it in the same directory as Models.py to before running it.

Also, the output about loss diagrams is uploaded in: https://drive.google.com/drive/folders/1pwEPe3AoYzYWHFDV-4hVpxA2SqOJPd00?usp=sharing

the output about training result is uploaded in: https://drive.google.com/drive/folders/1X4N8n4yuJw8QgVd51Av1dH5Pr-z5KKrq?usp=sharing

## Overview
Predicting total sales for every product and store in the next month using different ML algorithms

**input** - daily historical sales data and a test set
(*Note: list of shops and products slightly changes every month*)

**output** - total amount of products sold in every shop for the test set

## File Directories
data - contains all data used in prediction (this is uploaded to google drive)

output - the loss function diagram in each epoch from prediction (this would be generated when running the code)

result - the final result of prediction

root - all the python files user to perform prediction
- **ConvolutionalApproach.py** the convolutional approach running script
- **dataProcessing.py** perform all the data preprocessing and data exploration
- **linearModelApproach.py** the linear neural network approach running script
- **lstmApproach.py** the LSTM network approach running script
- **Models.py** contains all models used in the running scripts (lstm, convolutional, linear)
- **plotting.py** performs plotting for losses in each epoch during training
- **dataExploration.py** performs data exploration on training, test and submission data

## Technology Dependencies
| Technology name | Version |
| --------------- | ------- |
| pandas | 1.0.5 |
| numpy | 1.18.5 |
| Keras | 2.4.3 |
| matplotlib | 3.2.2 |

## Instruction of Running
1. Intall all dependencies by typing into command line:
    `pip3 install pandas`
    `pip3 install numpy`
    `pip3 install Keras`
    `pip3 install matplotlib`
2. Run data exploration by typing into command line `python3 dataExploration.py`. This should print out the result in command line
3. Run Convolutional Approach by typing into command line `python3 ConvolutionalApproach.py`. This should generate an output in "result/convolution_predicts.csv"
4. Run LSTM Approach by typing into command line `python3 lstmApproach.py`. This should generate an output in "result/lstm_predicts.csv"
5. Run Linear Network Approach by typing into command line `python3 linearModelApproach.py`. This should generate an output in "result/linear_predicts.csv"

Note that when running the scripts, it will also create a folder caller output and generate the loss diagram for each epochs in the output folder.

You can terminate the process anytime by hitting Ctrl + C

## Data
### File descriptions:
- **sales_train.csv** - the training set. Daily historical data from January 2013 to October 2015.
- **test.csv** - the test set. You need to forecast the sales for these shops and products for November 2015.
- **sample_submission.csv** - a sample submission file in the correct format.
- **items.csv** - supplemental information about the items/products.
- **item_categories.csv**  - supplemental information about the items categories.
- **shops.csv** - supplemental information about the shops.

### Files:
#### sales_train.csv
| varaible name | description |
| ------------- | ----------- |
| date | date in format dd/mm/yyyy|
| date_block_num | a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33 |
| shop_id | unique identifier of a shop |
| item_id | unique identifier of a product |
| item_price | current price of an item |
| item_cnt_day | number of products sold. You are predicting a monthly amount of this measure |

#### test.csv
| varaible name | description |
| ------------- | ----------- |
| ID | an Id that represents a (Shop, Item) tuple within the test set |
| shop_id | unique identifier of a shop |
| item_id | unique identifier of a product |

#### sample_submission.csv
| varaible name | description |
| ------------- | ----------- |
| ID | an Id that represents a (Shop, Item) tuple within the test set |
| item_cnt_day | number of products sold. You are predicting a monthly amount of this measure |

#### items.csv
| varaible name | description |
| ------------- | ----------- |
| item_name | name of item |
| item_id | unique identifier of a product |
| item_category_id | unique identifier of item category |

#### item_categories.csv
| varaible name | description |
| ------------- | ----------- |
| item_category_name | name of item category |
| item_category_id | unique identifier of item category |

#### shops.csv
| varaible name | description |
| ------------- | ----------- |
| shop_name | name of shop |
| shop_id | unique identifier of a shop |
