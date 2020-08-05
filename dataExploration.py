from dataProcessing import dataExploration, loadData, dataProcess

train, test, submission, items, itemCategory, shops = loadData()

dataExploration(train)
dataExploration(test)
dataExploration(submission)

dataset = dataProcess(train, test)