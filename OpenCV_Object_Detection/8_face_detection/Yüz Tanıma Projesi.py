# =============================================================================
#                             -Doğukan Karataş-                               #
# =============================================================================

import cv2
import matplotlib.pyplot as plt

#--------------------------------------------Einstein---------------------------------------------

# Resmi içe aktarıyoruz.
einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

# Sınıflandırıcıyı tanımlıyoruz.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # Eğitilmiş bir yüz sınıflandırıcısı

# Yüzü bir dikdörtgen içerisine alıyoruz.
face_rect = face_cascade.detectMultiScale(einstein)

# dikdörgen üzerindeki koordinatlar ve genişlik yükseklik parametreleri ile dikdötrgen boyut ve özelliklerini belirliyoruz.
for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")


#-------------------------------------------------------------------------------------------------

#-------------------------------------------Barcelona---------------------------------------------

# Resmi içe aktarıyoruz.
 
barce = cv2.imread("barcelona.jpg", 0)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")

face_rect = face_cascade.detectMultiScale(barce, minNeighbors = 7)
                                                # minNeighbors = 7 --> Ortak kutucuk sayısı ile en az hata elde ediliyor.

for (x,y,w,h) in face_rect:
    cv2.rectangle(barce, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")


#-------------------------------------------Video Kamera------------------------------------------
# Video kamerasını açıyoruz.

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read() # capture okundukça bize bir frame return edilecek
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
            
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),10)
        cv2.imshow("face detect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

# Video kameradan çıkış işlemleri    
cap.release()
cv2.destroyAllWindows()





























