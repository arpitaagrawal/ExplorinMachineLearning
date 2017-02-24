import kMeans
import numpy as np

# Test distance metric method
#a = kMeans.cal_dist_metric(np.random.normal(size=(10,3)), np.random.normal(size=(2,3)))
#print np.argmin(a, axis=1)

d = [1,2]
transformed_feature = np.empty([3])
transformed_feature[0] = d[0] ** 2
transformed_feature[1] = np.sqrt(2)*d[0]*d[1]
transformed_feature[2] = d[1] ** 2

print transformed_feature