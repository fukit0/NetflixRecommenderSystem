# (1)  when the argument is a numpy array, np.sum ultimately calls add.reduce to do the work.
#      The overhead of handling its argument and dispatching to add.reduce is why np.sum is slower.

import time

import numpy as np
import pandas
import scipy.stats.stats as pearsonr

startTime = time.time()

columnNames = ['movieID', 'userID', 'rating']

data = pandas.read_csv('TrainingRatings.txt', names=columnNames,
                       dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})

# gets first column for movie ids and second for user ids
listOfMovieIDs = data.movieID.tolist()
listOfUserIDs = data.userID.tolist()

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


def buildWeightMatrixBetweenUsers( ):
	for activeUserIndex in range(0, numberOfUsers - 1):
		print(str(activeUserIndex) + "th active user..")
		for otherUserIndex in range(activeUserIndex + 1, numberOfUsers):
			w = calculatePearsonCorrelation(activeUserIndex, otherUserIndex)
			# print ("w: " + str(w))
			userWeightMatrix[activeUserIndex, otherUserIndex] = w
			userWeightMatrix[otherUserIndex, activeUserIndex] = w

	return

############################################ END OF FUNCTIONS DEFINITIONS ########################################

# kullanıcılar tarafından herbir filme verilen puanlar tutuluyor
buildMovieUserRatingMatrix(data)
print("Movie-user matrix build time is %s" % (time.time() - startTime))

buildWeightMatrixBetweenUsers()

print(userWeightMatrix[0:10, 0:10])
print("Execution time: %s seconds." % (time.time() - startTime))
