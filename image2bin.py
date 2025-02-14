         
#----ref: https://076923.github.io/posts/Python-opencv-10/
                                                        
import cv2

#----------------------------------                                            
# png files to binary

src = cv2.imread("images/color.png", cv2.IMREAD_COLOR)
#file size
size=src.shape
size

#gray로 변환
gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

cv2.imshow("dst", dst)
# save as file
cv2.imwrite('images/gray.png', dst)


#----------------------------------
# Edge 검출
src = cv2.imread("images/color.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(src, 100, 255)
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

cv2.imshow("canny", canny)
cv2.imshow("sobel", sobel)
cv2.imshow("laplacian", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------
# image open and save

import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt

# Read image
im = pilimg.open("images/color.png")
im.show()
 
# Fetch image pixel data to numpy array
pix = np.array(im)
plt.imshow(pix)


#----------------------------------
# add a gaussian noise
# ref: https://stackoverflow.com/questions/22937589/how-to-add-noise-
#gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0 #pepper
            elif rdn > thres:
                output[i][j] = 255 #salt
            else:
                output[i][j] = image[i][j]
    return output

image = cv2.imread("images/bw.png", cv2.IMREAD_GRAYSCALE) # Only for grayscale image
noise_img = sp_noise(image, 0.1)
cv2.imwrite("sp_noise2.png", noise_img)


#----------------------------------
# add a text message on image
#

import cv2                       

# addtext.shape 명령어를 통해 text 위치결정(294,395)
# 영점은 좌측상단
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (150,280)
fontScale = 1
fontColor = (255,0,0)
lineType               = 2

addtext = cv2.imread('sp_noise1.jpg', 1)
cv2.putText(addtext,"0.1",bottomLeftCornerOfText, 
            font, fontScale, fontColor, lineType)
cv2.imwrite('add_text.jpg', addtext)
    
                                                                    




