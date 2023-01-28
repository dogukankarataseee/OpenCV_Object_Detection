# =============================================================================
#                             -Doğukan Karataş-                               #
# =============================================================================

import cv2
import numpy as np
from collections import deque # tesbit edilen objenin merkezini depolamak için

# nesne merkezini depolayacak veri tipi
buffer_size = 16 # deque boyutu 
pts = deque(maxlen = buffer_size) # nesnenin merkez pointleri

# mavi renk aralığı HSV
blueLower = (0,  99,  0)   # hsv = ton doygunluk parlaklık
blueUpper = (328, 255, 255)

# capture
cap = cv2.VideoCapture(0) # kameradan görüntü alma
cap.set(3,960) # 480x460 piksel görüntü elde etmek için
cap.set(4,480)

while True:
    
    success, imgOriginal = cap.read() # kameradan gelen veriyi oku
    
    if success: 
        
        # blur detayı azaltıp gurultuyu azaltmamız lazım 
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) #pencere boyutu ve standart sapma 
        
        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image",hsv)
        
        # mavi için maske oluştur
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("mask Image",mask)
       
        # maskenin etrafında kalan gürültüleri sil
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        cv2.imshow("Mask + erozyon ve genisleme",mask)
       
        # kontur
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            
            # en buyuk konturu al
            c = max(contours, key = cv2.contourArea)
            
            # dikdörtgene çevir 
            rect = cv2.minAreaRect(c)
            
            
            ((x,y), (width,height), rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            #print(s)
            
            # kutucuk
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # moment Görüntünün merkezini bulmamıza yarayan fonksiyon
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            # konturu çizdir: sarı
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255),2)
            
            # merkere bir tane nokta çizelim: pembe
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1)
            
            # bilgileri ekrana yazdır
            cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
            
        # deque ile ufak  bir takip algoritması
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(imgOriginal, pts[i-1], pts[i],(0,255,0),3)  
            
        cv2.imshow("Orijinal Tespit",imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break   # kameradan cıkmak için kullandıgımız komut
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
