# (1)  when the argument is a numpy array, np.sum ultimately calls add.reduce to do the work.
#      The overhead of handling its argument and dispatching to add.reduce is why np.sum is slower.

import time

import numpy as np
import pandas
import scipy.stats.stats as pearsonr
import buildUserWeightMatrix

startTime = time.time()

columnNames = ['movieID', 'userID', 'rating']

trainData = pandas.read_csv('TrainingRatings.txt', names=columnNames,
							dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})

# gets first column for movie ids and second for user ids
listOfMovieIDs = trainData.movieID.tolist()
listOfUserIDs = trainData.userID.tolist()

# unique movie IDs
# pandas.unique is faster then numpy.unique and return_index= true by default
uniqueListOfMovieIds = pandas.unique(listOfMovieIDs)
numberOfMovies = uniqueListOfMovieIds.size
tempMovieEnum = dict(enumerate(uniqueListOfMovieIds))
movieEnum = {v: k for k, v in
             tempMovieEnum.items()}  # inverse tempMovieEnum because enumerate does not return ordered way

# unique user IDs
uniqueListOfUserIDs = pandas.unique(listOfUserIDs)
numberOfUsers = uniqueListOfUserIDs.size
tempUserEnum = dict(enumerate(uniqueListOfUserIDs))
userEnum = {v: k for k, v in tempUserEnum.items()}  #inverse tempUserEnum because enumerate does not return ordered way

# movie X user matrix
movieUserRatingMatrix = np.empty(shape=(numberOfMovies, numberOfUsers), dtype=np.float)

# user X user weight matrix (r)
userWeightMatrix = np.empty(shape=(numberOfUsers, numberOfUsers), dtype=float)

############################################## FUNCTIONS DEFINITIONS #############################################

# First value is the r-value, 2nd is the p-value
def scipyPearsonrTest( index1, index2 ):
	print(pearsonr.pearsonr(movieUserRatingMatrix[:, index1], movieUserRatingMatrix[:, index2]))
	#print(pearsonr.pearsonr(index1, index2))
	return


def buildMovieUserRatingMatrix( data ):
	for index, row in data.iterrows():
		movieIndex = movieEnum[row['movieID']]
		userIndex = userEnum[row['userID']]
		movieUserRatingMatrix[movieIndex][userIndex] = row['rating']

	return


def getMeanRatingByUserIndex( activeUserIndex ):
	sum = 0
	numberOfRating = 0
	for rating in movieUserRatingMatrix[:, activeUserIndex]:
		if rating != 0:  # izlemediği filmleri hesaba katmıyoruz
			numberOfRating += 1
			sum += rating

	return sum / numberOfRating


def sumOfSquares( arr ):
	return sum([arr[i] ** 2 for i in range(0, arr.size)])


def getRatingsForBothUsers( activeUserVector, otherUserVector ):
	x = []
	y = []
	for i in range(0, activeUserVector.size):  ## both vectors have same size
		if activeUserVector[i] == 0 or otherUserVector[i] == 0:
			continue
		else:
			x.append(activeUserVector[i])
			y.append(otherUserVector[i])

	return np.asarray(x), np.asarray(y)

def calculatePearsonCorrelation( activeUserIndex, userIndex ):
	x, y = getRatingsForBothUsers(movieUserRatingMatrix[:, activeUserIndex], movieUserRatingMatrix[:, userIndex])
	mx = x.mean()
	my = y.mean()
	xm, ym = x - mx, y - my
	r_num = np.add.reduce(xm * ym)  # numpy.add.reduce equals to numpy.sum (1)
	r_den = np.sqrt(sumOfSquares(xm) * sumOfSquares(ym))
	r = r_num / r_den

	return r


def buildWeightMatrixBetweenUsers():
	for activeUserIndex in range(0, numberOfUsers-1):
		print(str(activeUserIndex) + "th active user..")
		for otherUserIndex in range(activeUserIndex + 1, numberOfUsers):
			w = calculatePearsonCorrelation(activeUserIndex, otherUserIndex)
			# print ("w: " + str(w))
			userWeightMatrix[activeUserIndex, otherUserIndex] = w
			userWeightMatrix[otherUserIndex, activeUserIndex] = w

	return

# Predict rating for user <userIndex>, for movie <movieIndex>
def predictAndCompareUserRating(testData,weightMatrix,knn):

	predictedMovies = np.zeros([len(testData)], dtype=[('movieID', '|S10'), ('userID', '|S10'), ('predictedRating', 'f4')])
	predictionCounter = 0
	recommendations = np.zeros([numberOfMovies], dtype=[('movieID', '|S10'), ('userID', '|S10')])
	recommendationCounter = 0
	maeValue = 0


	for index, row in testData.iterrows():
		movieID = row['movieID']
		movieIndex = movieEnum[movieID]
		userID = row['userID']
		userIndex = userEnum[userID]
		expectedRating = row['rating']

		meanRatingOfActiveUser = np.asarray(movieUserRatingMatrix[:,userIndex]).mean()

		numerator = 0
		denumerator = 0

		if userIndex < 100:
			# diğer userlar ile olan ağırlıkları büyükten küçüğe sıralarnı ve knn kadarı alınır
			sortedMostSimilarUsers = np.argsort(weightMatrix[userIndex][:])[::-1][:knn]

			for similarUserIndex in range(0, knn):
				mostSimilarUserIndex = sortedMostSimilarUsers[similarUserIndex]

				meanRatingOfOtherUser = np.asarray(movieUserRatingMatrix[:, mostSimilarUserIndex]).mean()
				otherUserRatingForMovie = movieUserRatingMatrix[movieIndex][mostSimilarUserIndex]

				numerator += (otherUserRatingForMovie - meanRatingOfOtherUser) * weightMatrix[userIndex][
					mostSimilarUserIndex]
				denumerator += weightMatrix[userIndex][mostSimilarUserIndex]

			predictedRating = meanRatingOfActiveUser + numerator / denumerator
			predictedMovies[predictionCounter] = ((str(movieID), str(userID), predictedRating))
			predictionCounter += 1
			maeValue += predictedRating - expectedRating


			if predictedRating > 4:
				recommendations[recommendationCounter] = (str(movieID), str(userID))
				recommendationCounter += 1


	np.savetxt('PredictRatings.txt', predictedMovies,delimiter=',',newline='\n', fmt='%s,%s,%f')
	np.savetxt('RecommendMovie.txt', recommendations, delimiter=',',newline='\n', fmt='%s,%s,%f')
	print("MAE : " + str(maeValue/predictionCounter))
	return

############################################ END OF FUNCTIONS DEFINITIONS ########################################

# kullanıcılar tarafından herbir filme verilen puanlar tutuluyor
buildMovieUserRatingMatrix(trainData)
print("Movie-user matrix build time is %s" % (time.time() - startTime))

# kullanıcılar arasındaki ağırlık matrisi oluşturulur
#buildWeightMatrixBetweenUsers()
weightMatrix = buildUserWeightMatrix.buildWeightMatrixBetweenUsers(movieUserRatingMatrix,5000)
#print(weightMatrix[0:10, 0:10])
print("Execution time: %s seconds." % (time.time() - startTime))


# read testRatings
testData = pandas.read_csv('TestingRatings.txt', names=columnNames, dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})
#print(len(testData))
predictAndCompareUserRating(testData,weightMatrix,5)
