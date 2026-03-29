import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. AYARLAR VE DOSYA YOLU ---
image_path = "data/BCCD_Dataset-master/BCCD/JPEGImages/BloodImage_00000.jpg"

if not os.path.exists(image_path):
    print(f"HATA: {image_path} bulunamadı!")
else:
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # --- 2. GÖRÜNTÜ İYİLEŞTİRME (Kritik Adım) ---
    # Kontrastı artırmak için CLAHE uyguluyoruz
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    blurred = cv2.medianBlur(enhanced_bgr, 5)

    # --- 3. AKYUVAR (WBC) TESPİTİ ---
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 30, 30]) # Aralığı biraz daha genişlettik
    upper_purple = np.array([175, 255, 255])
    mask_wbc = cv2.inRange(hsv, lower_purple, upper_purple)
    cleaned_wbc = cv2.morphologyEx(mask_wbc, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    wbc_contours, _ = cv2.findContours(cleaned_wbc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_output = img_rgb.copy()
    wbc_count = 0
    for cnt in wbc_contours:
        if cv2.contourArea(cnt) > 800:
            wbc_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(final_output, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(final_output, "WBC", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- 4. ALYUVAR (RBC) TESPİTİ (Watershed Agresif Ayar) ---
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Eşiklemeyi daha hassas yapıyoruz
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Eşiği 0.1'e kadar çektik. Bu en küçük zirveleri bile sayar.
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers)

    rbc_count = 0
    for label in np.unique(markers):
        if label <= 1: continue 
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 30: # En küçük hücreleri bile alıyoruz
                rbc_count += 1
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(final_output, (int(cx), int(cy)), 3, (255, 0, 0), -1)

    # --- 5. SONUÇLAR ---
    print("\n" + "="*40)
    print(f"AKYUVAR (WBC) SAYISI: {wbc_count}")
    print(f"ALYUVAR (RBC) SAYISI: {rbc_count}")
    print("="*40)

    plt.figure(figsize=(12, 7))
    plt.imshow(final_output)
    plt.title(f"Final: {wbc_count} WBC | {rbc_count} RBC")
    plt.axis('off')
    plt.show()