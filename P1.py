##### This part of the code will demo the Euler-Maruyama Methd. #####

from   numpy.random import Generator, PCG64 
from   numpy.random import Generator, PCG64
import numpy             as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

# Euler--Maryuma scheme
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

num_sims = 5  # Display five runs

t_init = 0
t_end  = 1
N      = 200  # Compute 1000 grid points
h      = (t_end - t_init) / N
x_init = -5.91652
y_init = -5.52332
z_init = 24.57231


def b1(x, y):
    # = 10(X(t)-Y(t))
    return 10 * (y - x)

def b2(x, y, z):
    # = X(t)(28-Z(t))-Y(t)
    return x*(28-z)-y

def b3(x, y, z):
    # = (X(t)Y(t)-8Z(t)/3)
    return (x*y-8*z/3)
    

def kesai():
    if np.random.binomial(1,0.5)==1:
        return 1
    else:
        return -1

ts = np.arange(t_init, t_end+h, h)
xs = np.zeros(N+1)
ys = np.zeros(N+1)
zs = np.zeros(N+1)

xs[0] = x_init
ys[0] = y_init
zs[0] = z_init


for i in range(1, ts.size):
    x = xs[i - 1]
    y = ys[i - 1]
    z = zs[i - 1]
    r = kesai()
    xs[i] = x + b1(x, y) * h + math.sqrt(h)*r
    r = kesai()
    print(xs[i])
    ys[i] = y + b2(x, y, z) * h + math.sqrt(h)*r
    r = kesai()
    zs[i] = z + b3(x, y, z) * h + math.sqrt(h)*r
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.plot(xs, ys, zs)
plt.title('Euler-Maruyama')
ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
plt.show()
print(xs)
#plt.plot(ts, xs)
#plt.plot(ts, ys)
#plt.plot(ts, zs)
#plt.xlabel("time (s)")
#h = plt.ylabel("y")
#h.set_rotation(0)
#plt.show()


def gen():
    return 1 if np.random.random() < 0.5 else -1

def eulermar(T, n): 
    h = T/n                         # step size

    xlist = np.zeros(n)
    ylist = np.zeros(n)
    zlist = np.zeros(n)
    xlist[0] = -5.91652 
    ylist[0] = -5.52332
    zlist[0] = 24.57231

    iter = n

    k = 1
    while k < iter:
        sqdt = np.sqrt(h)               # standard deviation of dU 
        ks = gen()

        xlist[k] = xlist[k-1] + h * (10*(ylist[k-1] - xlist[k-1])) + sqdt*ks
        ylist[k] = ylist[k-1] + h * (xlist[k-1]*(28-zlist[k-1]) - ylist[k-1]) + sqdt*ks
        zlist[k] = zlist[k-1] + h * (xlist[k-1]*ylist[k-1] - 8*zlist[k-1]/3) + sqdt*ks
        k = k + 1 

    return (xlist, ylist, zlist)

def methodB(T, n):
    h = T/n
    xlist = np.zeros(n)
    ylist = np.zeros(n)
    zlist = np.zeros(n)
    xlist[0] = -5.91652
    ylist[0] = -5.52332
    zlist[0] = 24.57231
    iter = 20 

    return

def methodC(T, n):
    h = T/n
    xlist = np.zeros(n)
    ylist = np.zeros(n)
    zlist = np.zeros(n)
    xlist[0] = -5.91652
    ylist[0] = -5.52332
    zlist[0] = 24.57231
    iter = n

    # while k < iter


##################################################################################

# set up initial value
n = 200
T = 10
xt, yt, zt = eulermar(T, n)


