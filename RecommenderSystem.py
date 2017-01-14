# (1)  when the argument is a numpy array, np.sum ultimately calls add.reduce to do the work.
#      The overhead of handling its argument and dispatching to add.reduce is why np.sum is slower.

import time

import numpy as np
import pandas
import scipy.stats.stats as pearsonr
import buildUserWeightMatrix # C'ye compile ettiğimiz dosyayı import ediyoruz

knn = 50
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

def buildMovieUserRatingMatrix( data ):
	for index, row in data.iterrows():
		movieIndex = movieEnum[row['movieID']]
		userIndex = userEnum[row['userID']]
		movieUserRatingMatrix[movieIndex][userIndex] = row['rating']

	return

# returns root mean square error value
def rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())


def getMeanRating(ratings):

	ratingArr = []

	for r in range(0,len(ratings)):
		if r > 0:
			ratingArr.append(r)

	return  np.asarray(ratingArr).mean()

# Predict rating for user <userIndex>, for movie <movieIndex>
def predictAndCompareUserRating(testData,weightMatrix,knn):

	predictedMovies = np.zeros([len(testData)], dtype=[('movieID', '|S10'), ('userID', '|S10'), ('predictedRating', 'f4')])
	predictionCounter = 0
	recommendations = np.zeros([numberOfMovies], dtype=[('movieID', '|S10'), ('userID', '|S10')])
	recommendationCounter = 0
	maeValue = 0

	rmsePredictions = []
	rmseExpecteds = []

	for index, row in testData.iterrows():
		# test dosyasından  satır okunuyor
		movieID = row['movieID']
		movieIndex = movieEnum[movieID]
		userID = row['userID']
		userIndex = userEnum[userID]
		expectedRating = row['rating']

		meanRatingOfActiveUser = getMeanRating(movieUserRatingMatrix[:,userIndex])

		numerator = 0
		denumerator = 0

		if userIndex < 100:
			# diğer userlar ile olan ağırlıkları büyükten küçüğe sıralarnı ve knn kadar user alınır
			sortedMostSimilarUsers = np.argsort(weightMatrix[userIndex][:])[::-1][:knn]

			for similarUserIndex in range(0, knn):
				#benzer kullanıcı indexini al
				similarUserIndex = sortedMostSimilarUsers[similarUserIndex]

				#benzer kullanıcının verdiği oyların ortalaması
				meanRatingOfSimilarUser = getMeanRating(movieUserRatingMatrix[:, similarUserIndex])
				#benzer kullanıcının movieID idli filme verdiği oy
				similarUserRatingForMovie = movieUserRatingMatrix[movieIndex][similarUserIndex]

				numerator += (similarUserRatingForMovie - meanRatingOfSimilarUser) * weightMatrix[userIndex][
					similarUserIndex]
				denumerator += weightMatrix[userIndex][similarUserIndex]

			predictedRating = meanRatingOfActiveUser + numerator / denumerator
			predictedMovies[predictionCounter] = ((str(movieID), str(userID), predictedRating))
			predictionCounter += 1
			maeValue += abs(predictedRating - expectedRating) # farkın mutlak değeri
			# rmse hesaplaması için değerler dizilere ekleniyor
			rmsePredictions.append(predictedRating)
			rmseExpecteds.append(expectedRating)

			# film tahminleme
			if predictedRating > 4:
				recommendations[recommendationCounter] = (str(movieID), str(userID))
				recommendationCounter += 1


	np.savetxt('PredictRatings.txt', predictedMovies,delimiter=',',newline='\n', fmt='%s,%s,%f')
	np.savetxt('RecommendMovie.txt', recommendations, delimiter=',',newline='\n', fmt='%s,%s')
	print("MAE  : " + str(maeValue/predictionCounter))
	print("RMSE : " + str(rmse(np.asarray(rmsePredictions),np.asarray(rmseExpecteds))))
	return

############################################ END OF FUNCTIONS DEFINITIONS ########################################

# kullanıcılar tarafından herbir filme verilen puanlar tutuluyor
buildMovieUserRatingMatrix(trainData)
print("Movie-user matrix build time is %s" % (time.time() - startTime))

# kullanıcılar arasındaki ağırlık matrisi oluşturulur
# C ye çevrilerek programın hızlandırılması amaçlanmıştır
weightMatrix = buildUserWeightMatrix.buildWeightMatrixBetweenUsers(movieUserRatingMatrix,numberOfUsers)
#print(weightMatrix[0:10, 0:10])
print("Execution time: %s seconds." % (time.time() - startTime))


# read testRatings
testData = pandas.read_csv('TestingRatings.txt', names=columnNames, dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})
#print(len(testData))
predictAndCompareUserRating(testData,weightMatrix,knn)
# program execution time takes about 55 hours