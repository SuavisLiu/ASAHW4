from   numpy.random import Generator, PCG64 
from   numpy.random import Generator, PCG64
import numpy             as np
import random
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

def ran(h):
    x = random.uniform(0,1)
    if x < 1/6:
        return np.sqrt(3*h)
    elif 1/6 < x < 1/3:
        return - np.sqrt(3*h)
    else:
        return 0

def gen():
    return 1 if np.random.binomial(1,0.5) == 1 else -1

def eulermar(T, n):
    h = T/n
    xlist = np.zeros(n)
    ylist = np.zeros(n)
    zlist = np.zeros(n)

    xlist[0] = -5.91652
    ylist[0] = -5.52332
    zlist[0] = 24.57231

    k = 1
    iter = n
    while k < iter:
        sqdt = np.sqrt(h)
        ks = gen()

        x = xlist[k-1]
        y = ylist[k-1]
        z = zlist[k-1]
        
        xlist[k] = x + 10 * (y - x) * h + ks * sqdt
        ks = gen()
        ylist[k] = y + (x * (28 - z) - y) * h + ks * sqdt
        ks = gen()
        zlist[k] = z + (x*y - 8*z/3) * h + ks * sqdt
        k = k + 1
    
    return (xlist, ylist, zlist)

def rungekutta(T, n):
    h = T/n
    xlist = np.zeros(n)
    ylist = np.zeros(n)
    zlist = np.zeros(n)

    xlist[0] = -5.91652
    ylist[0] = -5.52332
    zlist[0] = 24.57231

    k = 1
    iter = n
    while k < iter:
        x = xlist[k-1]
        y = ylist[k-1]
        z = zlist[k-1]
        
        ks = ran(h)
        xx = x + 10 * (y - x) * h + ks
        ks = ran(h)
        yy = y + (x * (28 - z) - y) * h + ks
        ks = ran(h)
        zz = z + (x*y - 8*z/3) * h + ks

        xlist[k] = x + 0.5*(10*(y-x) + 10*(yy-xx)) * h + ran(h)
        ylist[k] = y + 0.5*(x*(28-z)-y + xx*(28-zz)-yy) * h + ran(h)
        zlist[k] = z + 0.5*(x*y-8*z/3 + xx*yy-8*zz/3) * h + ran(h)
        k = k+1

    return (xlist, ylist, zlist)

def second(T,n):
    h = T/n
    xlist = np.zeros(n)
    ylist = np.zeros(n)
    zlist = np.zeros(n)

    xlist[0] = -5.91652
    ylist[0] = -5.52332
    zlist[0] = 24.57231 

    k = 1
    iter = n
    while k < iter:
        x = xlist[k-1]
        y = ylist[k-1]
        z = zlist[k-1]

        text = [[-10, 10, 0, 28-z, -1, -x, y, x, -8/3]]
        text = np.array(text)
        bprime = text[0].reshape((3,3))
        print (bprime)
        xyz = np.array([[x],
                         [y],
                          [z]])
        print(xyz)
        prod = np.matmul(bprime,xyz)
        xlist[k] = x + h*10*(y-x) + ran(h) + 0.5*(prod[0]+10*(y-x))*h*ran(h)+0.5*(10*(y-x)*prod[0])*pow(h,2)
        ylist[k] = y + h*(x*(28-z)-y) +ran(h) + 0.5*(prod[1]+(x*(28-z)-y))*h*ran(h)+0.5*((x*(28-z)-y)*prod[1])*pow(h,2)
        zlist[k] = z + h*(x*y-8*z/3) +ran(h) + 0.5*(prod[2]+(x*y-8*z/3))*h*ran(h)+0.5*(x*y-8*z/3)*prod[2]*pow(h,2)
        k = k+1
    
    return (xlist, ylist, zlist)



############################### Main ###############################
n = 20000
T = 10
xt, yt, zt = rungekutta(T, n)

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.plot(xt, yt, zt)
plt.title('Runge Kutta')
ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
plt.show()


