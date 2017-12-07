from scipy.io import loadmat
from PIL import Image
import numpy as np
import math
from scipy.special import expit
from sklearn.linear_model import LinearRegression


sizeOfTrain=30 * 10 # 30 person faces , 10 sample each one
sizeOfTest=10*10     # 10 person faces , 10 sample each one
sizeOfImage=64*64

RMSE=[]

def loadFaces():
    faces=loadmat('faces')
    faces = faces['faces'].T.copy()
    faces = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    return faces


def getData(faces):
    featureVectors=[]
    classVectors=[]
    featureVectorSize = math.ceil(sizeOfImage/2)

    for sample in faces:
        face=sample.ravel()
        featureVectors.append(face[0:featureVectorSize])
        classVectors.append(face[featureVectorSize:])

    return featureVectors,classVectors


def normalizeData(X):
    normalizedX=[]
    mean=np.mean(X)
    variance=np.std(X)
    for i in range(len(X)):
        normalizedX.append(X[i]/255)
    return np.array(normalizedX)


def train(faces):
    trainFeatureVectors, trainClassVectors = getData(faces[0:sizeOfTrain])
    X =normalizeData(trainFeatureVectors)
    Y =normalizeData(trainClassVectors)
    X_T = np.transpose(X)
    X_T_X = np.matmul(X_T, X)
    invXTX = np.linalg.inv(X_T_X)
    invXTX_XT = np.matmul(invXTX, X_T)
    W = np.matmul(invXTX_XT, Y)
    return W



def modifyResult(result):
    '''
    mean = result.mean()
    var=result.var()
    modifiedResult=[]
    OldMax=result.max()
    OldMin=result.min()
    NewMax = 255
    NewMin = 0
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)

    maxValue = max(result)
    for value in result:
        nValue = (((value - OldMin) * NewRange) / OldRange) + NewMin

        modifiedResult.append(nValue)
   '''
    modifiedResult=[]
    maxValue = max(result)
    minValue=min(result)
    for value in result:
        nValue=(value-minValue)/(maxValue-minValue)
        modifiedResult.append((nValue)*255)

    return modifiedResult

def test(faces,W):
    testFeatureVectors, testClassVectors = getData(faces[sizeOfTrain:])

    for i in range(len(testFeatureVectors)):
        testX=normalizeData(testFeatureVectors[i])
        target = testClassVectors[i]
        predictionResult = np.matmul(testX, W)
        predictionResult=modifyResult(predictionResult)
        calulateRMSE(target,predictionResult)
        wholeImage = list(testFeatureVectors[i])
        for value in predictionResult:
            wholeImage.append(value)


        imageOut = Image.new("L", (64, 64))
        imageOut.putdata(wholeImage)
        imageOut.save('result'+str(i)+'.jpg')


def calulateRMSE(target,result):
    tse=[]
    sum=0
    for i in range(len(target)):
        sum+=math.pow(target[i]-result[i],2)
    mse=sum/len(target)
    rmse=math.sqrt(mse)
    RMSE.append(rmse)


def performWithScikit(faces):
    global RMSE
    RMSE.clear()

    X, Y = getData(faces[0:sizeOfTrain])
    testX, testY = getData(faces[sizeOfTrain:])
    LR = LinearRegression()
    LR.fit(X, Y)
    predictY = LR.predict(testX)

    for i in range(len(predictY)):

        wholeImage = list(testX[i])
        for value in predictY[i]:
            wholeImage.append(value)
        calulateRMSE(testY[i],predictY[i])
        imageOut = Image.new("L", (64, 64))
        imageOut.putdata(wholeImage)
        imageOut.save('scikit_result' + str(i) + '.jpg')

    return RMSE




faces=loadFaces()
W=train(faces)
test(faces,W)
minRMSE=min(RMSE)
minIndex=RMSE.index(minRMSE)

print("RMSE : ")
print("BEST PREDICTION : ")
print("Result Number %d" % minIndex )
print("With RMSE %d" % minRMSE )
print("MY RMSE "  )
print(RMSE)

scikitRMSE=performWithScikit(faces)
minScikitRMSE=min(scikitRMSE)
minScikitIndex=RMSE.index(minScikitRMSE)
print("SCIKIT BEST PREDICTION : ")
print("Result Number %d" % minScikitIndex )
print("With RMSE %d" % minScikitRMSE )
print("SCIKIT RMSE "  )
print(scikitRMSE)
