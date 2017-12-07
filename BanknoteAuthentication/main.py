from random import seed
from random import randrange
from csv import reader
from math import sqrt
from scipy.special import expit
import numpy as np

accList=[]
precList=[]
recList=[]
FScoreList=[]

def loadData(filename):
	dataset = list()
	with open(filename, 'r') as f:
		csv_reader = reader(f)
		for row in csv_reader:
			if not row:
				continue
			else:
				dataset.append(row)
	return dataset


def findDatasetMinMax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def normalizeDataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])

def crossValidationSplit(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def calculateRMSE(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		error = actual[i] - predicted[i]
		sq_error = error**2
		sum_error += sq_error
	mean_error = sum_error/float(len(actual))
	return (sqrt(mean_error))

def crossValidationEvaluate(dataset, n_folds, *args):
	folds = crossValidationSplit(dataset, n_folds)
	RMSE =[]
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = logisticRegression(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		rmse = calculateRMSE(actual, predicted)
		calculateEvaluationScores(predicted, actual)
		RMSE.append(rmse)
	return RMSE


def	classify(prediction):
	classes=[]
	for value in prediction:
		if value>=0.5:
			classes.append(1.0)
		else:
			classes.append(0)

	return classes

def calculateEvaluationScores(prediction, actual):
	global accList,precList,recList,FScoreList
	accuracy=0
	truePositive=0
	falsePositive=0
	trueNegative=0
	falseNegative=0
	classes=classify(prediction)
	for i in range(len(actual)):
		if classes[i]==actual[i]:
			accuracy+=1
		if actual[i]==1 and classes[i]==1:
			truePositive+=1
		if actual[i] == 0 and classes[i] == 1:
			falsePositive+=1
		if actual[i]== 1 and classes[i]==0:
			falseNegative+=1
		if actual[i] == 0 and classes[i] == 0:
			trueNegative+=1

	accList.append((accuracy/len(actual)*100))
	precision=truePositive/truePositive+falsePositive
	precList.append(precision)
	recall=truePositive/truePositive+falseNegative
	recList.append(recall)
	FScore=(2*precision*recall)/(precision+recall)
	FScoreList.append(FScore)

	print("accuracy : %s"% str(accuracy/len(actual)*100))
	print("true positive : %s" % str(truePositive))
	print("true negative : %s" % str(trueNegative))
	print("false positive : %s" % str(falsePositive))
	print("false negative : %s" % str(falseNegative))
	print("precision : %s" % str(precision))
	print("recall : %s" % str(recall))
	print("F1Score : %s" % str(FScore))
	print("---_-----_-------_------_----\n")


def predict(row, coefficients):
	h = coefficients[0]
	for i in range(len(row)-1):
		h += row[i]*coefficients[i+1]
	h=expit(h)

	return h


def getAvgTSE(TSE):
	np

def gradientDescent(train, l_rate):
	coef = [0.0 for i in range(len(train[0]))]
	delta=1
	epoch=0
	TSE=[]
	while delta>0.002:
		epoch+=1
		for row in train:
			y = predict(row, coef)
			delta = y - row[-1]
			TSE.append(delta)
			coef[0] = coef[0] - l_rate*delta
			for i in range(len(coef)-1):
				coef[i+1] = coef[i+1] - l_rate*delta*row[i]
	print("Number of epochs: %d" % epoch)
	return coef

def logisticRegression(train, test, l_rate):
	predictions =[]
	coef = gradientDescent(train, l_rate)

	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return(predictions)

def calaulateAvg(data):
	sum=0
	for value in data:
		sum+=value
	return sum/len(data)

seed(1)
filename = 'data_banknote_authentication.csv'
dataset = loadData(filename)
for i in range(len(dataset[0])):
	for row in dataset:
		row[i] = float(row[i].strip())

minmax = findDatasetMinMax(dataset)
normalizeDataset(dataset, minmax)
nFolds = 5
learningRate = 0.1

scores = crossValidationEvaluate(dataset, nFolds, learningRate)
print('List of RMSEs: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))

print("Total statistics for all folds : \n")
print("Avg accuracy % f" % calaulateAvg(accList))
print("Avg precision % f" % calaulateAvg(precList))
print("Avg recall % f" % calaulateAvg(recList))
print("Avg FScore % f" % calaulateAvg(FScoreList))
