#load dataset
# %% codeblock
import numpy as np
import sys
# %% codeblock
#basics
#store data into array and list is very slow bcs numpy use  fix Type
# numpy they use less memory and contigous memory
a = np.array([[1,2,3],[4,5,6]],dtype='int16')
print(a)
#dimension
a.ndim
#row and coloum
a.shape
#et Type
a.dtype
#getsize
a.itemsize
#gettotal getsize
a.nbytes
#changing
a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
print(a)
#gets specific [row,coloum]
a[1,5]
a[0,:]
#startindex:endindex:stepsize
a[0,1:-1:2]
#change
a[1,5] = 20
print(a)
#3d examples
d = np.array([[[1,2],[3,4],[5,6],[6,7]]])
print(d)
#get specific element
d[:,1,:]
d
#initalizing 0
np.zeros((2,3))
np.ones((4,2,2))
np.full((2,2),99)
np.full_like(a.shape,4)
#random decimal number
np.random.rand(4,2,3)
np.random.random_sample(a.shape)
#random integer
np.random.randint(-4,8, size=(3,3))
arr= np.array([1,2,3])
#repeat
r1=np.repeat(arr,3)
print(r1)
r2=np.repeat(arr,3, axis=0)
print(r2)
#matrix
output =  np.ones((5,5))
print(output)
z = np.zeros((3,3))
z[1,1]=9
print(z)
output[1:-1,1:-1] = z
#-1 is last element
print(output)
#aritmatic
a+2
#linear algebra
a= np.ones((2,3))
print(a)
b= np.full((3,2),2)
print(b)
#multipication
np.matmul(a,b)
#finde determinant
c =np.identity(3)
np.linalg.det(c)
#statistic
np.min(d)
np.sum(d)
#reorganizing
before = a
print(before)
after = before.reshape ((6,1))
print(after)
#stacking
s = np.array([1,2,3,4])
d = np.array([4,3,3,4])
np.vstack([s,d])
np.hstack([s,d])
#bollean
np.any(s>1,axis=0)
