#image compression
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.io import loadmat
from matplotlib import pyplot as plt
%matplotlib inline
data=loadmat(r'C:\Users\parth_airmgsr\Downloads\ex7data2.mat')
x=data['X']
fig, ax=plt.subplots(figsize=(12,8))
ax.scatter(x[:,0],x[:,1],s=50)

def findclosestcentroids(x,centroids): #assighn closest centroid
    m=x.shape[0]
    idx=np.zeros(m)
    k=centroids.shape[0]
    for i in range(m):
        mdist=10000000
        
        for j in range(k):
            dist=np.sum((x[i,:]-centroids[j,:])**2)
            if dist<mdist:
                mdist=dist
                idx[i]=j
    return idx            
            
centroid = np.array([[5,3], [2, 8], [588,56]])
idx=findclosestcentroids(x,centroid)
idx

def computecentroid(x,idx,k): #adhust centroid to avg
    n=x.shape[1]
    centroids=np.zeros((k,n))
    for i in range(k):
        centroids[i]=np.sum(x[np.where(idx==i),:]/len(x[np.where(idx==i)]),axis=1)
        
    return centroids    
centroid=computecentroid(x,idx,3)
print(centroid)
fig, ax=plt.subplots(figsize=(12,8))
ax.scatter(centroid[:,0],centroid[:,[1]])
ax.scatter(x[:,0],x[:,1],s=50,c=idx,cmap='Reds')

def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    
    for i in range(max_iters):
        idx = findclosestcentroids(X, centroids)
        centroids = computecentroid(X, idx, k)
    
    return idx, centroids
initialcentroids=np.array([[2,3],[5,8],[3,9]])
print(initialcentroids.shape)
idx, centroids = run_k_means(x, initialcentroids, 10)
centroids
def initcentroids(x,k): # initialise random test samples as centroids
    m,n=x.shape
    centroids=np.zeros((k,n))
    idx=np.random.randint(0,m,k)
    for i in range(k):
        centroids[i,:]=x[idx[i],:]
    return centroids
initcentroids(x,3)
array([[4.1590816 , 0.61720733],
       [1.12036737, 5.20880747],
       [4.1590816 , 0.61720733]])
bla=np.array(np.arange(10).reshape(5,2))**2 #miscllaneus
np.sum(bla[[0,2,3],:],axis=1)
np.sum((x[np.where(idx==2),:])**2,axis=1)
array([[1020.91438059,  121.83365991]])

#image compression

imagedata=loadmat(r'C:\Users\parth_airmgsr\Downloads\bird_small.mat')
a=imagedata['A']
a.shape

np.max(a)
255
a=a/255
a=np.reshape(a,((a.shape[0]*a.shape[1]),a.shape[2]))
print(a.shape)
initialcentroids=initcentroids(a,16)
initialcentroids
idx,centroids=run_k_means(a, initialcentroids, max_iters=10)
idx=findclosestcentroids(a,centroids)
idx.shape , centroids.shape
centroids
xrecovered=centroids[idx.astype(int),:]
xrecovered.shape, centroids.shape,idx.shape
xrecovered=np.reshape(xrecovered,(imagedata['A'].shape))
xrecovered.shape



                                                        #original
plt.imshow(imagedata['A'])

plt.imshow(xrecovered), xrecovered.shape                        # compressed
