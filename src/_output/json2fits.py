#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:02:29 2025

@author: Jia Wei Teh

This file converts json dictionary (default output) into fits for better data extraction and analysis
"""

from pathlib import Path


path2json = r'/Users/jwt/unsync/Code/Trinity/outputs/1e7_sfe030_n1e4/dictionary_test.json'

filename = r'output.fits'

path2output = Path(path2json).parent / filename

import json
from astropy.io import fits
from astropy.table import Table

# Step 1: Load JSON file
with open(path2json) as f:
    data_dict = json.load(f)



# Step 2: Convert dictionary to Astropy Table
table = Table(rows = data_dict)


# Check type because object does not work
# print(list(c.dtype for c in table.columns.values()))

for ii, dtype in enumerate(table.dtype.descr):
    print(f'{ii}: {dtype}')
    
    
    
for col in table.colnames:
    if table[col].dtype == object:
        for value in table[col]:
            print(value, col)
            break


print(f'outputs will be saved to {path2output}')


# table.remove_columns(['F_ram'])
# print(table['F_ram'])


# for col in table.colnames:
#     if table[col].dtype == object:
#         for 
#         print(table[col].data)
#         break




object_cols = [col for col in table.colnames if table[col].dtype == object]


table.remove_columns(object_cols)






print(f'removed the following columns... {object_cols}')




# a = np.array([0.4,1,2,3.32,4.231,5.3])

# b = np.array2string(a, separator=',')

# c = np.fromstring(b.strip('[]'), sep = ',')




# Step 3: Write the table to a FITS file
table.write(path2output, format='fits', overwrite=True)

print("FITS file created successfully.")




#%%




hdul = fits.open(path2output)

hdul.info()


data = hdul[1]

print(data.data
     )


