import numpy
import os
import re 
import sys
import matplotlib.pyplot as plt
from linreg import LinReg

def genData(count, dim, m, minx, maxx, errstdv=1):
    for i in range(count):
        x_ = [1]
        for j in range(dim):
            x_.append(numpy.random.rand()*(maxx-minx) + minx)
        x_.append(numpy.dot(m, numpy.array(x_)) + numpy.random.normal(scale=errstdv))
        yield x_

def getGaussianData(count, dim, minx, maxx, errstdv=1):
    pass

dim = 20
b = 20
m = numpy.array([b]+[.1+numpy.random.normal(scale=.05) for i in range(dim)])
start=0
stop=100
errstdv=5
training = numpy.matrix(list(genData(1000,dim,m,start,stop,errstdv)))

_x_ = numpy.matrix(training.transpose()[0:-1]).transpose()
y = numpy.matrix(training.transpose()[-1]).transpose()

B = LinReg.getCoef(_x_, y)
#print(m, B)

test = numpy.matrix(list(genData(100,dim,m,start,stop,errstdv)))
eggs = numpy.matrix(test.transpose()[0:-1])
why = numpy.array(test.transpose()[-1])
fig, ax = plt.subplots()
ax.scatter(numpy.array(numpy.dot(eggs.transpose(), B)), why)

shample = numpy.arange(why.min(),why.max(),1)
ax.plot(shample, shample)

ax.set(xlabel='calculated', ylabel='actual',
       title='Linear Regression')
ax.grid()
plt.show()
