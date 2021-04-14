from   numpy.random import Generator, PCG64 
from   numpy.random import Generator, PCG64
import numpy             as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

def gran(A,b,x,R):
    ARAx = np.dot(A.T,np.dot(np.dot(R,A),x))
    ARb = np.dot(A.T,np.dot(R,b))
    g = 2*(ARAx-ARb)
    return(g)

def gnorm(A,b,x):
    return 2*(np.dot(np.dot(A.T,A),x)-np.dot(A.T,b))

def sgd(n,N):
    
    A = np.random.normal(0, 1/n, [n,n])

    eiA,evA=np.linalg.eig(np.dot(A.T,A))
    bd = max(eiA)
    elp=1/bd
    bs = np.zeros((n,1))
    bs[0,0] = 1
    iter = []
    
    for k in range(1,N+1):
    
        x = np.random.normal(0,1,[n,1])
        r = np.random.randint(0, high=n, size=1, dtype='l')
        R = np.zeros((n,n))
        R[r,r]=n

        grad0 = gran(A,bs,x,R)
        gradn0 = gnorm(A,bs,x)


        x = x - elp * grad0

        for j in range(2,30000): 
            # And then I will set the matrix R     
            # compute the G
            r = np.random.randint(0, high=n, size=1, dtype='l')
            R = np.zeros((n,n))
            R[r,r]= n
            grad = gran(A,bs,x,R)
            x = x - elp * grad/j
            gradn = gnorm(A,bs,x)
            if np.linalg.norm(gradn)/np.linalg.norm(gradn0) < 0.5:
                break
        if j!=29999:
            iter.append(j)
    return(np.mean(iter))
    
it=[]
n = []
for k in range(1,120+1,3):
    it.append(sgd(k,20))
    n.append(k)

print(it)