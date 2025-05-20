import cv2
import numpy as np

# 读取图片
img = cv2.imread('dog.jpg')

# 将图片调整为32x32大小
resized_img = cv2.resize(img, (32, 32))

# 显示原始图片和调整大小后的图片
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原始图片')
plt.axis('off')

plt.subplot(1, 2, 2) 
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
plt.title('调整大小后的图片')
plt.axis('off')

plt.tight_layout()
plt.show()
