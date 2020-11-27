import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc
import sys
import os
from scipy.fftpack import fft

import matplotlib.pylab as pylab
%matplotlib inline
pylab.rcParams['figure.figsize']=(20.0,7.0)


#to read the image
Input_image=cv.imread("F:/sem5/dip/forgery project/test_img.jpg",0)
f=plt.figure()
window_name='imagefirst'
plt.imshow(Input_image, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()


#define the functions for dct
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


#to divide the image into blocks
imsize=Input_image.shape
dct=np.zeros(imsize)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( Input_image[i:(i+8),j:(j+8)] )                             


#to take one block from the entire blocks(this is an example to show how the block is being converted in dct)
pos=128

plt.figure()
plt.imshow(Input_image[pos:pos+8,pos:pos+8], cmap='gray')
plt.title("An 8x8 image block")

plt.figure()
plt.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
plt.title( "An 8x8 DCT block")

#showing the entire blocks of dct transformed image
plt.figure()
plt.imshow(dct,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image")

# Threshold
thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))


plt.figure()
plt.imshow(dct_thresh,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "Thresholded 8x8 DCTs of the image")

percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)

print ( percent_nonzeros*100.0)


# comparing the original image and dct transformed image which actually seems to be the same
im_dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        im_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )
        
        
plt.figure()
plt.imshow( np.hstack( (Input_image, im_dct) ) ,cmap='gray')
plt.title("Comparison between original and DCT compressed images" )
