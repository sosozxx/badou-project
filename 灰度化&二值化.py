import numpy as  np
import matplotlib.pyplot as plt

import cv2
#灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
image_gray_float = np.zeros([h,w],img.dtype)
image_gray_ave  = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        image_gray_float[i,j] = (m[0] * 0.11 + m[1] * 0.59 +m[2]*0.3)
print(image_gray_float)
cv2.imshow("grayf",image_gray_float)
cv2.waitKey(1000)
# cv2.destroyWindow("grayf")

for i in range(h):
    for j in range(w):
        m = img[i,j]
        image_gray_ave[i,j] = int((m[0]  + m[1]  +m[2])//3)
print(image_gray_ave)
cv2.imshow("graya",image_gray_ave)
cv2.waitKey(1000)

img_gray_fun = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img_gray_fun)
cv2.imshow("gray",img_gray_fun)
cv2.waitKey(1000)

#二值化
img_binary = np.where(image_gray_float >=128,1,0)
print(img_binary)
print(img_binary.shape)

ret,img_binary_fun = cv2.threshold(img_gray_fun,127,255,cv2.THRESH_BINARY)
print(f"阈值{ret}")

plt.subplot(221)
plt.imshow(image_gray_float,cmap="gray")
plt.subplot(222)
plt.imshow(img_gray_fun,cmap="gray")
plt.subplot(223)
plt.imshow(img_binary,cmap="gray")
plt.subplot(224)
plt.imshow(img_binary_fun,cmap="gray")

plt.show()