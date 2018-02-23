
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pickle

f = open('w_final.pckl', 'rb')
w_final = pickle.load(f)
f.close()

f1 = open('b_final.pckl', 'rb')
b_final = pickle.load(f1)
f1.close()


# In[3]:


X_test = np.zeros((3000, 20534))
with open('testdata.txt') as fp2:
    for line in fp2:
        vals = line.split()
        X_test[(int(vals[1])-1), (int(vals[0])-1)] = float(vals[2])


# In[4]:


test = np.matmul(w_final.T, X_test) + b_final


# In[5]:


test = test.reshape((20534,1))


# In[8]:


import csv

test2 = []
print (test)

for i in test:
    test2.append([int(i[0])])
    
print (test2)

with open('out.csv', 'wb') as fh:
    writer = csv.writer(fh, delimiter=',')
    writer.writerow(['val'])
    
    
    writer.writerows(test2)

