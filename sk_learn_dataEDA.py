### Machine learning tutorial on Sci Kit learn from Data camp; tutorial can be found here : https://www.datacamp.com/community/tutorials/machine-learning-python ###
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the data 
digits = datasets.load_digits()

# Display the data set 
print(digits)

# Explore the data set 
print(digits.keys())

# Print data 
print(digits.data)

# Print out the target values
print(digits.target)

# Print ouf the description
print(digits.DESCR)

## Because this data is numpy array; the data is not sorted in an excel sheet format. It is important to know about the shape of the array to accurately handle the data. ##
# WE must seperate the data into a more usuable format #

# Isolate the digits data 
digits_data = digits.data

# Inspect
print(digits_data.shape)

# Isolate target values with the function target
digits_target = digits.target

# Inspect the shape of the new variable
print(digits_target.shape)

# Print the number of unique labels 
number_digits = len(np.unique(digits.target))

# Isolate the 'images' variable
digits_images = digits.images

# Inspect the shape of the digits set
print(digits_images.shape)

# You notice in the results that in the images set they are 8px by 8px. However, the target data has 64 features according to our inspection. We can reshape the data in order to visualize it. #
print(np.all(digits.images.reshape((1797,64)) == digits.data))

### Now we can vizualize the data; using Matplotlib ###

# Create the figure size you want (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjusdt the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

## For each of the 64 images created we modify the following ##
for i in range(64):
    
    # Initialize the subplots: add a subplot in the grid of 8 by 8 at the i+1-th position
    ax = fig.add_subplot(8,8, i + 1, xticks=[], yticks=[])

    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value 
    ax.text(0,7, str(digits.target[i]))

# Show the master plot
plt.show()






