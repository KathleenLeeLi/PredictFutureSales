from dataProcessing import dataExploration, loadData, dataProcess
from Models import createConvolutionalModel

train, test, submission, items, itemCategory, shops = loadData()

dataExploration(train)
dataExploration(test)
dataExploration(submission)

dataset = dataProcess(train, test)