#Import numpy module as np
import numpy as np
import matplotlib.pyplot as plt

'''
Ploting some numbers
'''
#Example 1
y = [4, 1, 2, 5, 8.7]
x = range(1, len(y)+1)
plt.plot(x, y)
plt.xlabel('I am x axis')
plt.ylabel('I am y axis')
plt.title('Example 1')
plt.show()

#Example 2
#Changing axis range
y = [4, 1, 2, 5, 8.7]
x = range(1, len(y)+1)
plt.plot(x, y)
plt.xlabel('I am x axis')
plt.ylabel('I am y axis')
plt.title('Example 2')
plt.axis([0, 6, 0, 10])
plt.show()

#Example 3
#Changing color and form
y = [4, 1, 2, 5, 8.7]
x = range(1, len(y)+1)
plt.plot(x, y, 'r-.')
plt.xlabel('I am x axis')
plt.ylabel('I am y axis')
plt.title('Example 3')
plt.axis([0, 6, 0, 10])
plt.show()

#Example 4
#Plotting multiple lines with legend
max_val = 5.
t = np.arange(0., max_val+0.5, 0.5)
plt.plot(t, t, 'r-', label='linear')
plt.plot(t, t**2, 'b--', label='quadratic')
plt.plot(t, t**3, 'g-.', label='cubic')
plt.plot(t, 2**t, 'y:', label='exponential')
plt.xlabel('I am x axis')
plt.ylabel('I am y axis')
plt.title('Example 4')
plt.legend()
plt.axis([0, max_val, 0, 2**max_val])
plt.show()

#Example 5
#Subplot
plt.title('Example 5')

ax = plt.subplot("221")
ax.set_title("Linear")
ax.plot(t, t, 'r-')
ax.set_ylabel('I am y axis')
ax.axis([0, max_val, 0, max_val])

ax = plt.subplot("222")
ax.set_title("Quadratic")
ax.plot(t, t**2, 'b--')
ax.axis([0, max_val, 0, max_val**2])

ax = plt.subplot("223")
ax.set_title("Cubic")
ax.plot(t, t**3, 'g-.')
ax.set_xlabel('I am x axis')
ax.set_ylabel('I am y axis')
ax.axis([0, max_val, 0, max_val**3])

ax = plt.subplot("224")
ax.set_title("Cubic")
ax.plot(t, 2**t, 'y:')
ax.set_xlabel('I am x axis')
ax.axis([0, max_val, 0, 2**max_val])

plt.show()

'''
Plotting with categorical variables
'''
#Example 6
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(1, figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
