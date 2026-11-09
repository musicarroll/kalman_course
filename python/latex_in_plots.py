# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:58:52 2023

@author: cognotrend
"""

import matplotlib.pyplot as plt
import numpy as npyghb 

# Enable LaTeX rendering in Matplotlib
plt.rcParams["text.usetex"] = True

# Create a NumPy array
arr = np.array([1.234567, 2.345678, 3.456789])

# Convert the array to a LaTeX string
latex_str = r'$\{' + ',\ '.join([f'{x:.2f}' for x in arr]) + r'\}$'

# Plot with LaTeX title
plt.plot([1, 2, 3], [4, 5, 6])
plt.title(latex_str)

plt.show()

