import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# assign directory
directory = '/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/Ourimage'
 
# iterate over files in
# that directory

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

count = 0

new_data_label = np.empty([40])
new_data = np.empty([40,64,64])

for filename in os.listdir(directory):
    if filename[0] == 'c':
        new_data_label[count] = 40
    elif filename[0] == 'f':
        new_data_label[count] = 41
    elif filename[0] == 'w':
        new_data_label[count] = 42
    elif filename[0] == 'x':
        new_data_label[count] = 43

    f = os.path.join(directory, filename)
  
    img = Image.open(f)
    img = img.resize((64,64),Image.ANTIALIAS)
    data = np.array(img)
    gray_data = rgb2gray(data)
    new_data[count,:,:]= gray_data

    count += 1

np.save("new_data.npy", new_data)
np.save("new_data_label", new_data_label)



