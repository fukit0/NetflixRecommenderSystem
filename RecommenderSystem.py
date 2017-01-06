import math as math

import numpy as np
import pandas
import scipy.stats.stats as pearsonr

columnNames = ['movieID', 'userID', 'rating']

data = pandas.read_csv('TrainingRatingsSmall.txt', names=columnNames,
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


############################################## FUNCTIONS DEFINITIONS #############################################

# First value is the r-value, 2nd is the p-value
def scipyPearsonrTest( index1, index2 ):
	print(pearsonr.pearsonr(movieUserRatingMatrix[:, index1], movieUserRatingMatrix[:, index2]))
	return

def buildMovieratingMatrix( data ):
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


def calculatePearsonCorrelation( activeUserIndex, userIndex ):
	numerator = 0
	denumeratorForActiveUser = 0
	denumeratorForUser = 0
	denumerator = 0

	meanRatingGivenByActiveUser = getMeanRatingByUserIndex(activeUserIndex)
	meanRatingGivenByUser = getMeanRatingByUserIndex(userIndex)

	for movie in movieUserRatingMatrix:
		ratingGivenByActiveUser = movie[activeUserIndex]
		ratingGivenByUser = movie[userIndex]

		numerator += (ratingGivenByActiveUser - meanRatingGivenByActiveUser) * (
		ratingGivenByUser - meanRatingGivenByUser)
		denumeratorForActiveUser = + math.pow(ratingGivenByActiveUser - meanRatingGivenByActiveUser, 2)
		denumeratorForUser = + math.pow(ratingGivenByUser - meanRatingGivenByUser, 2)

	denumerator = (math.sqrt(denumeratorForActiveUser)) * (math.sqrt(denumeratorForUser))

	# r = pearson correlation
	r = numerator / denumerator
	print(r)
	scipyPearsonrTest(activeUserIndex, userIndex)

	return


############################################ END OF FUNCTIONS DEFINITIONS ########################################

# kullanıcılar tarafından herbir filme verilen puanlar tutuluyor
buildMovieratingMatrix(data)

calculatePearsonCorrelation(0,1)
