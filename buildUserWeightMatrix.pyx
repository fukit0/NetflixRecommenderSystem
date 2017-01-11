from __future__ import division
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
            w = calculatePearsonCorrelation(movieUserRatingMatrix,activeUserIndex, otherUserIndex, numberOfUsers)
            # print ("w: " + str(w))
            userWeightMatrix[activeUserIndex, otherUserIndex] = w
            userWeightMatrix[otherUserIndex, activeUserIndex] = w

    return userWeightMatrix


def calculatePearsonCorrelation(np.ndarray[FLOAT_DTYPE_T,ndim=2] movieUserRatingMatrix, int activeUserIndex,int userIndex , int numberOfUsers):

    cdef float mx,my
    cdef x = []
    cdef y = []
    cdef xm = []
    cdef ym = []

    cdef float r,r_num,r_den

    x, y = getRatingsForBothUsers(movieUserRatingMatrix[:, activeUserIndex], movieUserRatingMatrix[:, userIndex])
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.add.reduce(xm * ym)  # numpy.add.reduce equals to numpy.sum (1)
    r_den = np.sqrt(sumOfSquares(xm) * sumOfSquares(ym))
    if r_den == 0:
        return 0

    return r_num / r_den


def getRatingsForBothUsers(np.ndarray[FLOAT_DTYPE_T, ndim=1] activeUserVector, np.ndarray[FLOAT_DTYPE_T, ndim=1] otherUserVector):

    cdef x = []
    cdef y = []
    cdef int arrIndex = 0

    ## both vectors have same size
    for i in range(0, activeUserVector.size):
        if activeUserVector[i] != 0 and otherUserVector[i] != 0:
            x.append(activeUserVector[i])
            y.append(otherUserVector[i])

    return np.asarray(x), np.asarray(y)


def sumOfSquares(np.ndarray[FLOAT_DTYPE_T, ndim=1] arr ):
    cdef int i

    return sum([arr[i] ** 2 for i in range(0, arr.size)])