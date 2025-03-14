import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ------------------------------------------------------------------------------
# 1) KULLANICI AYARLARI:
# Aşağıdaki yolları kendi bilgisayarınızdaki dizinlere göre düzenleyin.
# Örneğin:
#   C:\Users\alpip\face-analysis-app\dog-breed-backend\images
#   C:\Users\alpip\face-analysis-app\dog-breed-backend\annotations
#   C:\Users\alpip\face-analysis-app\dog-breed-backend\cropped_images
#   C:\Users\alpip\face-analysis-app\dog-breed-backend\stanford_dogs_model.h5
# ------------------------------------------------------------------------------
IMAGES_DIR = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\images'
ANNOTATIONS_DIR = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\annotations'
CROPPED_DIR = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\cropped_images'
MODEL_SAVE_PATH = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\stanford_dogs_model.h5'

# Eğitime devam edilecek toplam epoch sayısı
EPOCHS = 10

# ------------------------------------------------------------------------------
# 2) BİR RESMİ KIRPAN İŞLEV (Multithread ile her resmi işlemek):
#    Annotation dosyaları UZANTISIZ ama XML içeriği var.
# ------------------------------------------------------------------------------
def process_image(breed_folder, img_name):
    """
    Belirtilen breed klasöründeki bir resmi, annotation dosyasındaki bounding box
    bilgilerine göre kırpar ve cropped_images altındaki uygun klasöre kaydeder.
    """
    breed_path = os.path.join(IMAGES_DIR, breed_folder)
    annotation_breed_path = os.path.join(ANNOTATIONS_DIR, breed_folder)
    cropped_breed_path = os.path.join(CROPPED_DIR, breed_folder)
    
    os.makedirs(cropped_breed_path, exist_ok=True)
    
    img_path = os.path.join(breed_path, img_name)
    base_name = os.path.splitext(img_name)[0]  # Örn. n02085936_233
    annotation_file = os.path.join(annotation_breed_path, base_name)  # Uzantı eklemiyoruz!

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"Resim okunamadı: {img_path}")
        return

    if not os.path.exists(annotation_file):
        print(f"Annotation dosyası bulunamadı: {annotation_file}")
        return

    try:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
    except Exception as e:
        print(f"XML parse hatası: {annotation_file}, {e}")
        return

    h, w, _ = img_cv.shape
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    if xmax <= xmin or ymax <= ymin:
        print(f"Geçersiz bounding box: {img_path}")
        return

    cropped_img = img_cv[ymin:ymax, xmin:xmax]
    if cropped_img.size == 0:
        print(f"Kırpılmış resim boş: {img_path}")
        return

    save_path = os.path.join(cropped_breed_path, img_name)
    cv2.imwrite(save_path, cropped_img)

# ------------------------------------------------------------------------------
# 3) TÜM RESİMLERİ MULTITHREAD İLE KIRP VE KAYDET
# ------------------------------------------------------------------------------
def create_cropped_dataset_multithread():
    """
    images/ klasöründeki her cinsin resimlerini, annotation dosyalarındaki bounding box bilgisine
    göre paralel (multithread) olarak kırpar ve cropped_images/ klasörüne kaydeder.
    """
    os.makedirs(CROPPED_DIR, exist_ok=True)
    
    breed_folders = [d for d in os.listdir(IMAGES_DIR)
                     if os.path.isdir(os.path.join(IMAGES_DIR, d))]
    
    with ThreadPoolExecutor(max_workers=13) as executor:
        futures = []
        for breed_folder in breed_folders:
            print(f"İşleniyor: {breed_folder}...")
            breed_path = os.path.join(IMAGES_DIR, breed_folder)
            image_files = [f for f in os.listdir(breed_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in image_files:
                futures.append(executor.submit(process_image, breed_folder, img_name))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Hata oluştu:", e)
    
    print("Tüm kırpma işlemleri tamamlandı!")

# Önce dataset kırpma aşamasını çalıştırıyoruz (her seferinde tekrar koşar ama önceden kırpılanlar değişmez)
create_cropped_dataset_multithread()

# ------------------------------------------------------------------------------
# 4) KIRPILMIŞ GÖRÜNTÜLERLE MODEL EĞİTİMİ (Fine-Tuning Dahil)
# ------------------------------------------------------------------------------
def train_model():
    """
    Cropped_images klasörünü kullanarak, MobileNetV2 tabanlı bir modeli köpek cinsi sınıflandırması için
    eğitim ve doğrulama seti ile eğitir. Eğer daha önce eğitilmiş bir model varsa, onu yükleyip
    ince ayar (fine-tuning) yapar.
    """
    # --- Daha güçlü veri artırma (augmentation) ayarları ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory=CROPPED_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    print("Eğitim verisi class_indices:", train_generator.class_indices)

    val_generator = val_datagen.flow_from_directory(
        directory=CROPPED_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Toplam sınıf sayısı (köpek cinsi): {num_classes}")

    # --- Mevcut modeli yükle veya yeni model oluştur ---
    if os.path.exists(MODEL_SAVE_PATH):
        print("Önceden kaydedilmiş model bulundu, yüklüyor...")
        model = load_model(MODEL_SAVE_PATH)
        # Modeli ince ayar yapmak için son birkaç blok katmanını açalım
        # (Bu örnekte "block_14" ve "block_15" ismindeki katmanlar. Adı mobilenetv2'de geçebilir.)
        base_model = model.layers[0]  # Mobilenetv2 tabanı ilk layer olarak varsayılmış durumda
        
        unfreeze = False
        for layer in reversed(base_model.layers):
            if 'block_14' in layer.name or 'block_15' in layer.name:
                unfreeze = True
            if unfreeze:
                layer.trainable = True

        # Yeni optimizer (düşük learning rate) ile recompile
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    else:
        print("Kaydedilmiş model yok, sıfırdan oluşturuluyor...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        for layer in base_model.layers:
            layer.trainable = False  # ilk aşama için tüm katmanları donduruyoruz
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # --- Callback'ler: EarlyStopping ve ModelCheckpoint ---
    checkpoint_callback = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,  # 3 epoch boyunca val_accuracy iyileşmezse durdur
        restore_best_weights=True
    )

    # --- Eğitimi başlat ---
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,  # max 10 epoch
        callbacks=[checkpoint_callback, early_stop]
    )

    print(f"Model kaydedildi (veya güncellendi): {MODEL_SAVE_PATH}")

# Model eğitim fonksiyonunu çağır
train_model()

print("Tüm işlem tamamlandı!")
