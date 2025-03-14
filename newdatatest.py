import os
import random
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ayarlar - Lütfen yolları kendi ortamınıza göre düzenleyin.
MODEL_PATH = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\stanford_dogs_model.h5'
MAPPING_PATH = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\mapping.json'
# Test veri dizini; alt klasörler içinde resimler var.
TEST_DIR = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\images.cv_b0q06o473ua8pspfvlqbwu\data\test'

def load_mapping(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping

def collect_all_test_images(test_dir):
    """
    test_dir altında alt klasörler varsa, her klasörden
    (örneğin "animal dog maltese", "animal dog german_sheperd", vb.)
    tüm resim dosyalarının yolunu ve alt klasör adını (true label olarak) toplar.
    """
    valid_ext = ('.jpg', '.jpeg', '.png')
    result = []
    subfolders = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    for sf in subfolders:
        folder_path = os.path.join(test_dir, sf)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            # Gerçek etiket: mapping.json'da tanımlı ise alt klasör ismi, aksi halde tüm klasör adını kullanabilirsiniz.
            result.append((img_path, sf))
    return result

def main_test_10():
    # Modeli ve mapping'i yükle
    model = load_model(MODEL_PATH)
    mapping = load_mapping(MAPPING_PATH)
    
    # Test resimlerini topla
    all_images = collect_all_test_images(TEST_DIR)
    total_imgs = len(all_images)
    if total_imgs == 0:
        print("Test dizininde resim bulunamadı!")
        return

    print(f"Test dizininde toplam {total_imgs} resim bulundu.")
    
    # Rastgele 10 resim seç (detaylı görmek için)
    random.shuffle(all_images)
    chosen = all_images[:100]
    print("\nRastgele seçilen 10 resim ve alt klasör (gerçek etiket):")
    for idx, (img_path, folder_name) in enumerate(chosen, start=1):
        print(f"{idx}. {img_path} (Klasör: {folder_name})")
    
    correct = 0
    total_confidence = 0.0
    print("\nTahminler:")
    for idx, (img_path, folder_name) in enumerate(chosen, start=1):
        # Gerçek etiket için mapping uygulayalım; mapping.json dosyanızda alt klasör adları varsa,
        # örneğin "animal dog maltese" şeklinde tanımlı ise onu alırız.
        true_label = mapping.get(folder_name, folder_name)
        
        # Resmi oku
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"{idx}. Resim okunamadı: {img_path}")
            continue
        
        # Ön işleme: Modelin beklediği 224x224 boyutuna resize, normalleştirme
        img_resized = cv2.resize(img_cv, (224, 224))
        img_arr = image.img_to_array(img_resized)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = img_arr / 255.0
        
        # Tahmin yap
        preds = model.predict(img_arr)
        pred_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(preds[0][pred_class])
        total_confidence += confidence
        
        # breed_dict'ten tahmin edilen etiketi alın (mapping anahtarları string olmalı)
        predicted_label = mapping.get(str(pred_class), "Unknown")
        
        # Basit eşleşme kontrolü: Tam eşitlik veya string containment
        is_correct = predicted_label.lower() == true_label.lower()
        if is_correct:
            correct += 1
        
        print(f"{idx}. Resim: {img_path}")
        print(f"   Gerçek etiket: {true_label}")
        print(f"   Tahmin: {predicted_label} (güven: {confidence:.2f})")
        print(f"   Sonuç: {'DOĞRU' if is_correct else 'YANLIŞ'}\n")
    
    accuracy = (correct / len(chosen)) * 100
    avg_confidence = total_confidence / len(chosen)
    print(f"Toplam 10 resim üzerinde basit doğruluk oranı: {accuracy:.2f}%")
    print(f"Ortalama güven oranı: {avg_confidence:.2f}")

if __name__ == "__main__":
    main_test_10()
