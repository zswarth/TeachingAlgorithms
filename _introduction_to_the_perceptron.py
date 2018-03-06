#### Introduction to the Perceptron. 
#### Fill In ALL QUESTIONS
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from numpy import genfromtxt

a = genfromtxt('test.csv', delimiter=',')
labels=a[:,0]
data = a[:,1:-1]

#b = python convert image to array

def img(row, data):
	image = np.zeros((28,28))
	for i in range(0,28):
		for j in range(0,28):
			pix = 28*i+j
			image[i,j] = data[row, pix+1]
	plt.imshow(image, cmap = 'gray')
	plt.show()
	print data[row,0]

# img(88, a) #this tells you what image is at a speciic data point

def one_number(labels, number):
	new_labels = []
	for i in range(0, len(labels)):
		if labels[i] == number:
			new_labels.append(1)
		else:
			new_labels.append(0)
	return new_labels
new_labels = one_number(labels, 7) 


#__________DAY 1______________

##Questoins

### 1) What is the difference between Supervised and Unspervised Learning
#Supervised learning uses previous data to learn what patterns to look for in future sets of data.
#Unsupervised learning figures out patterns in data by grouping similar data with similar data without previously provided information. 
### 2) What is the difference bewteen Regression and Classification Problems
#I DON'T GET THIS DEFINITION:
#In Regression problems, the data inputted will be continuous, meaning one piece of data at a time.  
#In Classification problems, the data inputted will be class labels, or, groups of data that sort the individual pieces of data into groups of analogous data.

#### Project

## We have two data sets.  All our work needs to be done for AN ARBITRARY NUMBER OF DIMENTIONS.  The firt Data set is in two dimentions.  The second is in 4 (This is the famous Iris Data Set)

#Data Set 1: 2D Data
# labels = np.array([0, 1,0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,0,1])
# data = np.array([[29,126,58,29,94,255,51,71,90,280,282,283,86,229,57,48,194,174,18,196,80,22,55,133,249,218,144,86,258,238,28,0,109,38,195,38,121],[120,51,158,176,83,88,143,161,116,57,116,92,173,59,167,183,24,74,107,109,181,195,167,66,47,27,02,07,20,29,155,137,102,135,58,147,1]])

# Labels_test = np.array([0,1,0,0,1,1,0,0,0,1,1])


# data_test = np.array([[90,194,1,24,195,111,53,90,99,193,208], [141,89,126,178,118,96,168,179,163,65,62]])

#Data Set 2: Iris FLower Data

#Make sure you have Iris.txt downloaded

#flowers = np.loadtxt('Iris.txt', dtype=str)
#flowers = np.asfarray(flowers[1:, :], dtype=float)

#We're going to work with this data set later.

# data = data.transpose()

def add_ones(x):
	#Write a funtion that addes a column of ones to the array
	a, b = np.shape(x)
	c = np.ones((a , b + 1))
	c[:,1:] = x
	return c
data = add_ones(data)


# print np.shape(data[12])
# print labels[12]

### rewrite our test data with a column of ones

def graph(data, labels):
	for i in range (0, len(labels)):
		if labels[i] == 0:
			plt.scatter(data[i, 0], data[i, 1], color = "royalblue", marker = "*")
		if labels[i] == 1:
			plt.scatter(data[i, 0], data[i, 1], color = "teal", marker = "<")
	plt.show()
# graph(data, labels)
	#write a function that will plot the first two dimentions of the data. Make sure different shapes/colors are used for each label.
	#data is a two column arry of x and y
	#plt.scatter graph 
	# in range (first x coordinate, first y coordinate, marker = "symbol," color = "")
	#i is the row and 0 is the column 


#______________Day 2___________________

def create_weights(data):
	a, b = np.shape(data)
	weights = []
	for i in range (0, b):
		weights.append(rd.random())
	a = np.asarray(weights)
	return a

a = create_weights(data)
# print a
#write a funtion that will return a numpy array of RANDOM WEIGHTS.  The array should be the correct length for the data given.

def predict(data_point, weights):
	b = np.dot(data_point, weights)
	if b > 0:
		return 1
	else:
		return 0
# print predict(data[20], a)
# print "PREDICTION"
#to make it not a specific number in the data set, set up a loop for a range of i, as in data[i]
#predict function we did in class a couple days ago
#random number generator in python (import random) weights are random numbers 
#for any given data point, return the predicted value (shoudl be 0 or 1 for a binary classification)

# #_____________Day 3____________________

def update(weights, data_point, labels, alpha):
	predicted = predict(data_point, weights)
	for i in range(0, len(weights)):
		weights[i] = weights[i] - alpha*(predicted - labels)*data_point[i]
	return weights
# print update(a, data[0], labels[0], .2)

def train_perceptron(data, labels, weights, alpha = .01, iterations = 100):
	for i in range(0, iterations):
		for i in range(0, len(data)):
			weights = update(weights, data[i], labels[i], alpha)
		return weights
# weights =  train_perceptron(data, labels, a, .5, 100000)
# weights =  train_perceptron(data, new_labels, a, .1, 130) #this runs the perceptron with the data from test.csv
# print weights, 'weights'

# #______________Day 4_________________

def test_percepton(data, labels, weights):
	x = 0
	for i in range(0, len(data)):
		predicted = predict(data[i], weights)
		if predicted == labels[i]:
			x += 1
		percentage = (float(x) / len(data)) * 100
	return percentage, "Percentage Correct"
# print test_percepton(data, new_labels, weights)

def all_numbers(labels, data):
	c,d = np.shape(data)
	weights = np.zeros((10,d))
	for i in range(0, 10):
		z = one_number(labels, i)
		x = train_perceptron(data, z, a, .1, 20) #this is the iteration you need to change to make it work better
		# print x[0:10], 'alskdjf'
		weights[i] = x
	return weights
# print np.shape(all_numbers(labels, data))

# w = all_numbers(labels, data)
# print w[0:10]

# np.savetxt("final_weights.csv", w, delimiter=",")

w = np.genfromtxt("final_weights.csv", delimiter=",")

def one_all(data_point, weights):
	for i in range(0, 10):
		if predict(data_point, weights[i]) == 1:
			return i 
	return 0

def answer(data, w):
	for i in range(0, 20):
		final_answer = one_all(data[i], w) 
		compare = labels[i]
		print final_answer, compare 
print answer(data, w)

def test_function(data, w, labels):
	for i in range (0, 20):
		x = 0
		answered = one_all(data[i], w)
		if answered == labels[i]:
			x += 1
			percentage = (float(x) / 20) * 100 #Wrong number here where "20" is...what should it be?
		return percentage
print test_function(data, w, labels), "Percentage Correct"


