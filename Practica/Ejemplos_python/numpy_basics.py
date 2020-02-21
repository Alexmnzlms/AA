#Import numpy module as np
import numpy as np

d1 = np.zeros((9, ), np.int32)
d1 = d1.reshape((-1, 3))

print(d1)



'''
   Creating some numpy arrays
'''
#Filled with zeros
d1_zeros = np.zeros((5), np.float32) #1d float32 zeros array
d2_zeros = np.zeros((2, 3), np.int32) #2d int32 zeros array
d5_zeros = np.zeros([i for i in range(5, 26, 5)], np.uint8) #5d uint8 zeros array

#Filled with ones
d5_ones = np.ones(d5_zeros.shape, np.bool) #5d bool ones array

#From python list
list2d = [[1, 3], [4, 5]]
d2_array = np.array(list2d, np.int32) #2d int32 array

#From uniform distribution
u4d = np.random.uniform(size=(5, 5, 5, 5)) #5x5x5x5 uniform array [0, 1.0)
u2d = np.random.uniform(low=-2.5, high=5.0, size=(2, 2)) #2x2 uniform array [-2.5, 5.0)
u3d_int = np.random.random_integers(low=5, high=15,size=(5, 5, 6)) #5x5x6 uniform int array [5, 15]

#From normal (gaussian) distribution
#3x3x8 array with elements draw from a normal distribution with 0 mean and 5 std.
g3d = np.random.normal(loc=0.0, scale=5.0, size=(3, 3, 8))

'''
   Basic operations
'''
#Getting the shape of an array
print ('Shape', g3d.shape)

#Getting the type
print ('Type', g3d.dtype)

#Changing the shape
u1d = u2d.reshape(-1)

print ('u2d new shape', u1d.shape)

#Changing the type
u3d = u3d_int.astype(np.float32)

print ('u3d new type', u3d.dtype)

#Swap axis
print ('u3d_int swapped axis shape', u3d_int.swapaxes(2, 0).shape)

#Copy array
u2d_copy = u2d.copy()

#Element-wise operations
sum_cte = d2_zeros + 5
sum2d = u2d + d2_array
rest_cte = d2_zeros - 1
rest2d = u2d - d2_array
prod_cte = sum_cte*4
prod2d = u2d * d2_array
div_cte = sum_cte/2
div2d = u2d / d2_array
exp_cte = sum_cte**5
exp2d = u2d**d2_array

#Absolute value
u2d_abs = np.abs(u2d)

print ('Min u2d', u2d.min(), 'Min u2d_abs', u2d_abs.min())

#Transpose
print ('d2_zeros transpose shape', d2_zeros.transpose().shape)

#Matrix product
mprod2d = u2d.dot(sum_cte)

#Inverse
i1 = np.linalg.inv(np.eye(2))

#SVD
U, s, V = np.linalg.svd(u2d)

#Mean
u2d_mean = u2d.mean()
u2d_mean_rows = u2d.mean(axis=1)

print ('Mean:', u2d_mean, u2d_mean_rows)

#STD
u2d_std = u2d.std()
u2d_std_rows = u2d.std(axis=1)

print ('STD:', u2d_std, u2d_std_rows)

#Sum
u2d_sum = u2d.sum()
u2d_sum_colums = u2d.sum(axis=0)

print ('Sum:', u2d_sum, u2d_sum_colums)

#Save numpy array
np.save('u2d.npy', u2d)
u2d_loaded = np.load('u2d.npy')

print ('Diference between loaded u2d and u2d', np.abs(u2d-u2d_loaded).max() < 10**(-5))

'''
   Indexing
'''
print ('Array from 1 to 7 step 2', u4d.reshape(-1)[1:7:2])
print ('Array from 1 to end step 2', u4d.reshape(-1)[1::2])
print ('Array from ini to end step 2', u4d.reshape(-1)[::2])
print ('Array from 1 to 7 step 1', u4d.reshape(-1)[1:7:])

#Negative index
print ('Last element', u1d[-1], 'Element three before the last position', u1d[-3])

#Integer array indexing
print ('Selecting elemnts 1, 2 and 3', u1d[[1, 2, 3]])

#Bolean array Indexing
no_u2d_neg = u2d.copy()
no_u2d_neg[no_u2d_neg < 0] = 0

print ('u2d no negative elements min', no_u2d_neg.min())
