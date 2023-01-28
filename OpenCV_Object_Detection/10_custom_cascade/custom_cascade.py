# =============================================================================
#                             -Doğukan Karataş-                               #
# =============================================================================
"""
n = negative picture aranmayan
p = pozitive picture  aranan

"""


import cv2
import os

# resim depo klasörü
path = "images"

# resim boyutu
imgWidth = 180
imgHeight = 120

# video capture
cap = cv2.VideoCapture(0) # Default camera
cap.set(3, 640) 
cap.set(4, 480)
cap.set(10, 180) # Kameranın aydınlık seviyesi

# klasör oluşturma
global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    os.makedirs(path+str(countFolder))

saveDataFunc()

count = 0
countSave = 0

# Kamerayı aktif hale getirip veri topluyoruz.
while True:
    
    success, img = cap.read()
    
    if success:
        
        img = cv2.resize(img, (imgWidth, imgHeight))
        # Aynı örnekleri tekrar almaması için kontrol ediyoruz.
        if count % 5 == 0: # aynı 5 resimden 1 tanesini depola
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png",img)
            countSave += 1
            print(countSave)
        count += 1
        
        cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()


