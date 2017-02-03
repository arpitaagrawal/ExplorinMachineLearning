import numpy as np


a = np.random.normal(size=(10,3))
b = np.random.normal(size=(1,3))
print 'a', a
print 'b',b
print np.shape(np.sqrt(np.sum((a-b)**2,axis=1)))  