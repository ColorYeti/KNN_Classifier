'''
---K-Nearest-Neighbour implementation on Python 3.6---
Implemented by following http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

Predicts type of flower by classifying training data into classes and comparing that to the data and assigning it to the class nearest to the data
'''
import numpy as np
import csv
import random
import math
import operator

def LoadData(filename,split,trainingSet=[],testSet=[]):                              #Load the data, split it into training and test sets according to split ratio
    with open(filename,'rt') as csvfile:                                             #Open file as csv(Comma Seperated Values). 'rt' makes it open in read mode in text
        lines=csv.reader(csvfile)                                                    #returns an iterator object. each row in the iterator object contains the data extracted from the file 
        dataset=list(lines)                                                          #puts the data in the iterator object into a list
        for x in range(len(dataset)-1):                                              #x is the row of the data set
            for y in range(4):                                                       #y is the column of the dataset. y runs from 0 to 3 only those indexes contain the measurement data.
                dataset[x][y]=float(dataset[x][y])                                   #converts the numbers in text to float
                if random.random()<split:                                            #randomly selects rows to insert to training set
                    trainingSet.append(dataset[x])
                else:                                                                #randomly selects rows to insert to test set
                    testSet.append(dataset[x])

def EuclideanDistance(instance1,instance2,length):                                   #Straight line distance between two points using the equation sqrt(a^2+b^2+c^2+....+x^2)
    distance = 0                                                                     #initialise distance
    for x in range(length):                                                          #x is the index of measurement data
        distance+= pow((instance1[x]-instance2[x]),2)                                #implement equation(a1-a2)^2+(b1-b2)^2+...+(x1-x2)^2
    return math.sqrt(distance)                                                       #returns sqrt((a1-a2)^2+(b1-b2)^2+...+(x1-x2)^2)

def GetNeighbours(trainingSet,testInstance,k):                                       #Get k nearest neighbours
    distances=[]                                                                     #initialise array distances
    neighbours=[]                                                                    #initialise array neighbours
    length = len(testInstance)-1                                                     #get number of measurement data in instance
    for x in range(len(trainingSet)):                                                #x is index of column of data in set
        dist = EuclideanDistance(testInstance,trainingSet[x],length)                 #Get distance between training and distance set
        distances.append((trainingSet[x],dist))                                      #add distance into distances array
    distances.sort(key=operator.itemgetter(1))                                       #sort distances by the first distance in the array from smallest to greatest
    for x in range(k):                                                               #x is specified range k
        neighbours.append(distances[x][0])                                           #add k nearest neighbours into array neighbours
    return neighbours                                                                

def GetResponse(neighbours):                                                         #classify the the data into its class
    classVotes={}                                                                    #initialize dictionary classVotes
    for x in range(len(neighbours)):                                                 #x is row of neighbours
        response=neighbours[x][-1]                                                   #get last column(type of flower) of row x for neighbours and put in response
        if response in classVotes:                                                   #if the type of flower is a key in the dictionary classVotes
            classVotes[response]+=1                                                  #add 1 to the type of lower
        else:
            classVotes[response]=1                                                   #else create a new key in the dictionary and initialise it at value of 1
        sortedVotes = sorted(classVotes.items(),key=operator.itemgetter,reverse=True)#sort the dictionary from greatest to smallest
        return sortedVotes[0][0]                                                     #return the name of flower with greatest value ie. index 0

def GetAccuracy(testSet,predictions):                                                #Get the accuracy of the prediction
    correct=0                                                                        #initialise integer correct
    for x in range(len(testSet)):                                                    #x is column row of testSet
        if testSet[x][-1] is predictions[x]:                                         #if predictions and actual flower name match
            correct+=1                                                               #add 1 to correct
    return (correct/float(len(testSet)))*100                                         #accuracy is correct/total

#region TestBlock                                                                    #Test each function
def TestData():
    with open('iris.data','rt') as csvfile:
        lines=csv.reader(csvfile)
        for row in lines:
            print(', '.join(row))

def TestLoadData():
    trainingSet=[]
    testSet=[]

    LoadData('iris.data',0.66,trainingSet,testSet)
    print('Train: ' + repr(len(trainingSet)))
    print('Train: '  + repr(len(testSet)))

def TestEuclideanDistance():
    data1=[2,2,2,'a']
    data2=[4,4,4,'b']
    print('Distance: ' + repr(EuclideanDistance(data1,data2,3)))

def TestGetNeighbours():
    trainSet=[[2,2,2,'a'],[4,4,4,'b']]
    testInstance=[5,5,5]
    neighbours=GetNeighbours(trainSet,testInstance,3)
    print(neighbours)

def TestGetResponse():
    neighbours=[[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
    response=GetResponse(neighbours)
    print(response)

def TestGetAccuracy():
    testSet=[[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
    predictions=['a','a','a']
    accuracy=GetAccuracy(testSet,predictions)
    print(accuracy)
#endregion

def main():                                                                          #main function
    trainingSet=[]                                                                   #initialise array trainingSet
    testSet=[]                                                                       #initialise array testSet
    predictions=[]                                                                   #initialise array predicitons
    split=.95                                                                        #initaialise float split with the ratio that splits input data into training and test sets 
    k=5                                                                              #initialise integer k that specifies amount of neighbours to compare to

    LoadData('iris.data',split,trainingSet,testSet)                                  #load the data into trainingSet and testSet
    print('Train: '+repr(len(trainingSet)))                                          #print number of rows in trainingSet
    print('Test: '+repr(len(testSet)))                                               #print number of rows in testSet
    for x in range(len(testSet)):                                                    #x is the row of testSet
        neighbours=GetNeighbours(trainingSet,testSet[x],k)                           #Get k nearest neighbours
        results=GetResponse(neighbours)                                              #Get prediction of flower type
        predictions.append(results)                                                  #Add prediction into array predictions
        print('Predicted= '+repr(results)+'  Actual= '+repr(testSet[x][-1]))         #print the predicted and actual flower type
    accuracy=GetAccuracy(testSet,predictions)                                        #get the accuracy of predictions
    print('Accuracy: '+repr(accuracy)+'%')                                           #print the accuracy

main()                                                                               #executes main function
