import numpy as np

### Parameters
iter = 'L01'
delta_t = [40, 45]
eta = [0.002]
tau = [1000]
lamb = [0.9]
dur = '30'
b_l = '0.013'

f = open('jobs-learning.txt', 'w+')

for i in delta_t:
    for j in eta:
        for k in tau:
            for l in lamb:
                line = 'python code-learning.py' + ' ' + iter + ' ' + str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + str(l) + ' ' + dur + ' ' + b_l + '\n'
                f.write(line)
f.close()
