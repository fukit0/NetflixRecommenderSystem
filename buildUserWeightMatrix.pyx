
from libc.math cimport sqrt
import numpy as np
cimport numpy as np

FLOAT_DTYPE = np.float
ctypedef np.float_t FLOAT_DTYPE_T

def buildWeightMatrixBetweenUsers(np.ndarray[FLOAT_DTYPE_T,ndim=2] movieUserRatingMatrix, int numberOfUsers):

    cdef int activeUserIndex,otherUserIndex
    cdef float w
    cdef np.ndarray[FLOAT_DTYPE_T, ndim=2] userWeightMatrix = np.zeros([numberOfUsers,numberOfUsers], dtype=FLOAT_DTYPE)

    for activeUserIndex in range(0, numberOfUsers-1):
        print(str(activeUserIndex) + "th active user..")
        for otherUserIndex in range(activeUserIndex + 1, numberOfUsers):
            w = calculatePearsonCorrelation(movieUserRatingMatrix,activeUserIndex, otherUserIndex)
            # print ("w: " + str(w))
            userWeightMatrix[activeUserIndex, otherUserIndex] = w
            userWeightMatrix[otherUserIndex, activeUserIndex] = w

    return userWeightMatrix


def calculatePearsonCorrelation(np.ndarray[FLOAT_DTYPE_T,ndim=2] movieUserRatingMatrix, int activeUserIndex,int userIndex ):

    cdef float mx,my
    cdef np.array x,y,xm,ym
    cdef float r,r_num,r_den

    x, y = getRatingsForBothUsers(movieUserRatingMatrix[:, activeUserIndex], movieUserRatingMatrix[:, userIndex])
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)  # numpy.add.reduce equals to numpy.sum (1)
    r_den = np.sqrt(sumOfSquares(xm) * sumOfSquares(ym))
    r = r_num / r_den

    return r


def getRatingsForBothUsers(np.array activeUserVector,np.array otherUserVector ):

    cdef np.array x,y

    for i in range(0, activeUserVector.size):  ## both vectors have same size
        if activeUserVector[i] == 0 or otherUserVector[i] == 0:
            continue
        else:
            x.append(activeUserVector[i])
            y.append(otherUserVector[i])

    return np.asarray(x), np.asarray(y)


def sumOfSquares(np.array arr ):
    cdef int i

    return sum([arr[i] ** 2 for i in range(0, arr.size)])