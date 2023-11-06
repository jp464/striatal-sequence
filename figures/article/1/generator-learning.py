import numpy as np

f = open('jobs-learning.txt', 'w+')
for i in np.arange(40, 70, 10):
    for j in np.arange(0.04, 0.091, 0.01):
        i, j = int(i), round(j,2)
        f.write('python code-learning.py ' + 'learning-env ' + str(i) + ' ' + str(j) + ' 1000 0.6 01 20 0.02 ' + '\n')
f.close()
