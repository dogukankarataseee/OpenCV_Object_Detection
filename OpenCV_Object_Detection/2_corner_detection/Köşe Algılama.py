# =============================================================================
#                             -Doğukan Karataş-                               #
# =============================================================================

import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içe aktar
img = cv2.imread("sudoku.jpg", 0) # resmi içe siyah beyaz aktarma yapıyoruz.
img = np.float32(img)
print(img.shape)  # resmin boyutunu öğreniyoruz
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

# harris corner detection   komşuluk boyutu  kutucuk boyut
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

dst = cv2.dilate(dst, None) # genişletme
img[dst>0.2*dst.max()] = 2
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

# ==================== farklı bir detection methodu ====================

# shi tomasi detection
img = cv2.imread("sudoku.jpg", 0)      # resmi içe aktarma
img = np.float32(img)                  
corners = cv2.goodFeaturesToTrack(img, 100,0.01, 10) # köşe sayısı #iki koşe arası min uzaklık
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel() # düzleştir
    cv2.circle(img, (x,y),3,(125,125,125),cv2.FILLED) # köşelere daire eklenir.
     
plt.imshow(img)
plt.axis("off")