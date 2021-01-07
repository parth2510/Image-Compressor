
data=loadmat(r'C:\Users\parth_airmgsr\Downloads\ex7data1.mat')
x=data['X']
x.shape

fig, ax =plt.subplots(figsize=(12,8))
ax.scatter(x[:,0],x[:,1])
x.std(axis=0).shape
x.mean(axis=0)

def pca(x):
    x=x-x.mean(axis=0)/x.std(axis=0)
    cov=np.dot(x.T,x)/x.shape[0]
    u,s,v=np.linalg.svd(cov)
    return u,s,v
u,s,v=pca(x)

m=(u[1,1]-u[0,1])/(u[1,0]-u[0,0])
c=u[0,1]-(m*u[0,0])
x1=-1
x2=6
y1=m*x1+c
y2=m*x2+c
import matplotlib.lines as mlines ##
fig, ax =plt.subplots(figsize=(12,8))
ax.scatter(x[:,0],x[:,1])
ax.plot(u[:,0],u[:,1])
#ax.set_xlim([-20,20])
#ax.set_ylim([-20,20]) 
def projectdata(x,u,k):
    ureduced=u[:,:k]
    return np.dot(x,ureduced)
z=projectdata(x,u,1)
z,z.shape

fig,ax=plt.subplots(figsize=(12,8))
plt.plot(z[0],z[40],)

def recoverdata(z, u, k):
    ureduced = u[:,:k]
    return np.dot(z, ureduced.T)

xrecovered = recoverdata(z, u, 1)
xrecovered[:,:]

fig, ax=plt.subplots(figsize=(12,8))
ax.scatter(x[:,0],x[:,1],s=50)
ax.scatter(xrecovered[:,0],xrecovered[:,1],c='red')
#plt.plot(u[:,0],u[:,1])
plt.plot(u[0,0],u[0,1],marker='x')
############################################################################ to plot the line u
# compute coe 
fig, ax=plt.subplots(figsize=(12,8))
#ax.scatter(x[:,0],x[:,1],s=50)
ax.scatter(xrecovered[:,0],xrecovered[:,1],c='red')
z=projectdata(x,u,2)
xrecovered = recoverdata(z, u, 2)
fig, ax=plt.subplots(figsize=(12,8))
ax.scatter(x[:,0],x[:,1],s=100)
ax.scatter(xrecovered[:,0],xrecovered[:,1],c='red')


faces = loadmat(r'C:\Users\parth_airmgsr\Downloads\ex7faces.mat')
x = faces['X']
x.shape , x 

plt.imshow((x[0:10,:].reshape((32,32,10)).T))
