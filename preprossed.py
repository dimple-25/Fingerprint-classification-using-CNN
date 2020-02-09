#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
path = r'D:\IITI\final_database\New folder (3)'
files = os.listdir(path)

for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpeg'])))

