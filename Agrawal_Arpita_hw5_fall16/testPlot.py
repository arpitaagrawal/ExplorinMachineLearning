"""
Simple demo of a scatter plot.
"""
#from numpy import *
import matplotlib.pyplot as plt
import read_data
import kMeans
import math
import numpy as np
#read_data.read_data()

#kMeans.transform_data_using_kernel(read_data.blob_train_data)
# N = np.shape(read_data.blob_train_data)[0]
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = np.pi * (4)**2  # 0 to 15 point radiuses
# 
# plt.scatter(read_data.blob_train_data[:,0], read_data.blob_train_data[:,1], s=area, c=colors, alpha=0.5)
# plt.show()

# # covariance matrix
# sigma = matrix([[2.3, 0, 0, 0],
#            [0, 1.5, 0, 0],
#            [0, 0, 1.7, 0],
#            [0, 0,   0, 2]
#           ])
# # mean vector
# mu = array([2,3,8,10])
# 
# # input
# x = array([2.1,3.5,8, 9.5])
# 
# def norm_pdf_multivariate(x, mu, sigma):
#     size = len(x)
#     if size == len(mu) and (size, size) == sigma.shape:
#         det = linalg.det(sigma)
#         if det == 0:
#             raise NameError("The covariance matrix can't be singular")
# 
#         norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
#         x_mu = matrix(x - mu)
#         print 'shape(x_mu)', shape(x_mu)
#         inv = sigma.I
#         print 'shape(inv)', shape(inv) 
#         print 'shape(x_mu * inv * x_mu.T)', shape(x_mu * inv * x_mu.T)       
#         result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
#         
#         print 'shape(result)', shape(result)
#         return norm_const * result
#     else:
#         raise NameError("The dimensions of the input don't match")
# 
# print norm_pdf_multivariate(x, mu, sigma)
x1 = np.arange(9.0).reshape((3, 3))

print 'x1', x1
x2 = np.arange(9.0).reshape((3, 3))

print 'x2', x2
print '\n\n'

print np.multiply(x1, x2)
print '\n\n'


x = np.arange(9.)
print x
print 'test', x[np.where( x == 5 )]

x = np.array([(1.5,2,3), (4,5,6)], dtype=np.float64)

print '\nmax', np.argmax(x, axis =1)
print '\n',x
print '\n\n', np.mean(x,axis =0), '\n\n'
#avg_density = np.empty([2][3])
for i in range(0,2):
    for j in range(0,3):
        t = np.sum(x[i,])
        print x[i][j]/t
        x[i][j] = x[i][j]/t
        
print x


a = [3,6,9]

print "\n", np.shape(a)
print "\n", np.shape(np.array([a]))

b = np.array([a])
print b.shape
