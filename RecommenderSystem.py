import numpy as np
import pandas

columnNames = ['movieID', 'userID', 'rating']

data = pandas.read_csv('TrainingRatingsSmall.txt', names=columnNames,
                       dtype={'movieID': np.str, 'userID': np.str, 'rating': np.float})

# gets first column for movie ids and second for user ids
listOfMovieIDs = data.movieID.tolist()
listOfUserIDs = data.userID.tolist()

# unique movie IDs
uniqueListOfMovieIds = pandas.unique(
	listOfMovieIDs)  # pandas.unique is faster then numpy.unique and return_index= true by default
numberOfMovies = uniqueListOfMovieIds.size
tempMovieEnum = dict(enumerate(uniqueListOfMovieIds))
movieEnum = {v: k for k, v in tempMovieEnum.items()}

# unique user IDs
uniqueListOfUserIDs = pandas.unique(listOfUserIDs)
numberOfUsers = uniqueListOfUserIDs.size
tempUserEnum = dict(enumerate(uniqueListOfUserIDs))
userEnum = {v: k for k, v in tempUserEnum.items()}

# movie X user matrix
userMovieRatingMatrix = np.empty(shape=(numberOfMovies, numberOfUsers), dtype=np.float)


############################################## FUNCTIONS DEFINITIONS #############################################

def buildMovieratingMatrix( data ):
	for index, row in data.iterrows():
		movieIndex = movieEnum[row['movieID']]
		userIndex = userEnum[row['userID']]
		userMovieRatingMatrix[movieIndex][userIndex] = row['rating']

	return


############################################ END OF FUNCTIONS DEFINITIONS ########################################

# kullanıcılar tarafından herbir filme verilen puanlar tutuluyor
buildMovieratingMatrix(data)
