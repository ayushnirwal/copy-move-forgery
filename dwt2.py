import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
from matplotlib.colors import NoNorm
import cv2


import numpy as np
from scipy.spatial import distance

#initializing parameters

block_size = 10

mean_threshold = 0.000005
SD_threshold = 0.0000005
min_pixel_distance = 30
check_offset=3


#reading image and provided mask ( provided mask will be used for accuracy calulation)

img = mpimg.imread('forged1.png')
img = np.array(img)

mask= mpimg.imread('forged1_mask.png')
mask= np.array(mask)

#converting to grayscale

mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)




column = img_gray.shape[1] - block_size +1
row = img_gray.shape[0] - block_size +1

prediction_mask = np.zeros((img_gray.shape[0], img_gray.shape[1]))


data = []


#creating array of dict objects ( with mean, SD, i, j ) by calc mean and SD of every block

print("making blocks and calculating mean with block_size = " + str(block_size))

block_counter = 0
for i in range(0, row):
    for j in range(0, column):

        block = img_gray[i:i+block_size, j:j+block_size]


        d=dict()
        avg_mat[i][j]=np.mean(block)
        d['M'] = np.mean(block)
        d['SD'] = np.std(block)
        d['i'] =    i
        d['j'] =    j

        data.append(d)

        block_counter+=1
        
        
print("Done")

# sorting according to Mean
sorted_mean = sorted(data, key=lambda element: element['M']) 



# distinguishing similar blocks ( only checking neighbours in sorted array for potential similar blocks) 
sim_array=[]

for i in range(len(sorted_mean)):

    for j in range( max(0,i-check_offset), min( len(sorted_mean), i+check_offset) ):
        mean_similarity = abs(sorted_mean[j]['M'] - sorted_mean [i]['M'])
        SD_similarity = abs(sorted_mean[j]['SD'] - sorted_mean [i]['SD'])

        coor1 = np.array([ sorted_mean[i]['i'] , sorted_mean[i]['j']])
        coor2 = np.array([ sorted_mean[j]['i'] , sorted_mean[j]['j']])

        distance = np.linalg.norm(coor1-coor2)

        if mean_similarity <= mean_threshold and SD_similarity <= SD_threshold and distance >= min_pixel_distance:
            
            sim_array.append(sorted_mean[i])
            sim_array.append(sorted_mean[j])


print(len(sim_array))

#creating prediction mask from similar blocks

for ele in sim_array:
    i = ele['i']
    j = ele['j']
    prediction_mask [i:i+block_size, j:j+block_size] = 1



#creating overlay for visual representation of comparision b/w predicted mask and provided mask
#black => True negative
#white => True positive
#green => False positive
#red => False positive


overlay=np.zeros((img_gray.shape[0], img_gray.shape[1],3))


TP = 0
FP = 0
TN = 0
FN = 0

for i in range(0, prediction_mask.shape[0]):
    for j in range(0, prediction_mask.shape[1]):
        if prediction_mask[i][j] == mask_gray[i][j]:
            if prediction_mask[i][j] == 1:
                overlay[i][j] = [255,255,255]
                TP+=1
            else:
                overlay[i][j] = [0,0,0]
                TN+=1
        else:
            if prediction_mask[i][j] == 1:
                overlay[i][j] = [255,0,0]
                FP+=1
            else:
                overlay[i][j] = [0,255,0]
                FN+=1


precision = TP/(TP+FP)
recall = TP/(TP+FN)
accuracy = 2*precision*recall/(precision+recall)

print('Accuracy:', accuracy)


plt.figure(1)
plt.imshow(overlay,cmap="gray")


plt.show()






