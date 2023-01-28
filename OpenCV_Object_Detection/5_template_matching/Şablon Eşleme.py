# =============================================================================
#                             -Doğukan Karataş-                               #
# =============================================================================

import cv2
import matplotlib.pyplot as plt

# template matching: sablon esleme

img = cv2.imread("cat.jpg", 0)            # orijinal resmi içeri aktar.
print(img.shape)          # şekil boyutlarını öğreniyoruz.
template = cv2.imread("cat_face.jpg", 0)  # tesbip edilecek resmi içeri aktarıyoruz.
print(template.shape)     # şekil boyutlarını öğreniyoruz.
h, w = template.shape     # tespit edilecek resim boyutlarını değişkenlere aktarıyoruz.
                            #print(h)
                            #print(w)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# methodlarıfor döngüsünde deniyoruz.
for x in methods:
    
    method = eval(x) # 'cv2.TM_CCOEFF' -> cv2.TM_CCOEFF methodları string'den çıkarıp normal fonsiyon gibi kullanabiliyoruz.
    res = cv2.matchTemplate(img, template, method)
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(x)
    
    plt.show()
    
    
    
 
    
    
    
    