import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Dosya Yolu
image_path = "data/BCCD_Dataset-master/BCCD/JPEGImages/BloodImage_00000.jpg"

if not os.path.exists(image_path):
    print("Resim bulunamadi!")
else:
    # 2. Resmi Oku
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Ön İşleme (Gürültü Silme)
    blurred = cv2.medianBlur(img_bgr, 5)

    # 4. HSV Renk Uzayı ve Mor Filtre
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 40, 40]) 
    upper_purple = np.array([170, 255, 255])
    mask_wbc = cv2.inRange(hsv, lower_purple, upper_purple)

    # 5. Gürültü Temizleme (Küçük noktaları sil)
    kernel = np.ones((5,5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask_wbc, cv2.MORPH_OPEN, kernel)

    # 6. Konturları Bul ve Çiz
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img_rgb.copy()
    wbc_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: # Sadece büyük olanları (akyuvar) say
            wbc_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(output_img, f"WBC #{wbc_count}", (x, y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 7. TÜM SONUÇLARI TEK EKRANDA GÖSTER
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Orijinal Goruntu")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cleaned_mask, cmap='gray')
    plt.title("2. Temizlenmis WBC Maskesi")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(output_img)
    plt.title(f"3. Tespit Edilen: {wbc_count} WBC")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Terminal Bilgisi: Toplam {wbc_count} adet Akyuvar bulundu.")