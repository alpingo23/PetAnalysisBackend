import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# GPU Kullanımını Maksimize Et
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Kullanılıyor: {gpus}")
    except RuntimeError as e:
        print(e)

# Mixed Precision Kullanarak Eğitimi Hızlandır
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Kullanıcı ayarları (WSL2 için)
IMAGES_DIR = '/mnt/c/Users/alpip/face-analysis-app/dog-breed-backend/images'
ANNOTATIONS_DIR = '/mnt/c/Users/alpip/face-analysis-app/dog-breed-backend/annotations'
CROPPED_DIR = '/mnt/c/Users/alpip/face-analysis-app/dog-breed-backend/cropped_images'
MODEL_SAVE_PATH = '/mnt/c/Users/alpip/face-analysis-app/dog-breed-backend/stanford_dogs_model.tf'
PERIODIC_MODEL_PATH = '/mnt/c/Users/alpip/face-analysis-app/dog-breed-backend/stanford_dogs_epoch_{epoch:03d}.tf'

EPOCHS = 1000  # En az 10 saat çalışacak
BATCH_SIZE = 256  # RTX 3080 için optimize

# Resimleri kırpan fonksiyon
def process_image(breed_folder, img_name):
    breed_path = os.path.join(IMAGES_DIR, breed_folder)
    annotation_breed_path = os.path.join(ANNOTATIONS_DIR, breed_folder)
    cropped_breed_path = os.path.join(CROPPED_DIR, breed_folder)

    os.makedirs(cropped_breed_path, exist_ok=True)

    img_path = os.path.join(breed_path, img_name)
    base_name = os.path.splitext(img_name)[0]
    annotation_file = os.path.join(annotation_breed_path, base_name + '.xml')

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return

    if not os.path.exists(annotation_file):
        return

    try:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
    except:
        return

    h, w, _ = img_cv.shape
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)

    if xmax <= xmin or ymax <= ymin:
        return

    cropped_img = img_cv[ymin:ymax, xmin:xmax]
    if cropped_img.size == 0:
        return

    save_path = os.path.join(cropped_breed_path, img_name)
    cv2.imwrite(save_path, cropped_img)

# Çoklu iş parçacığı (Multithread) ile kırpma
def create_cropped_dataset_multithread():
    os.makedirs(CROPPED_DIR, exist_ok=True)

    breed_folders = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]

    with ThreadPoolExecutor(max_workers=16) as executor:  # Daha fazla thread aç
        futures = [executor.submit(process_image, breed_folder, img_name)
                   for breed_folder in breed_folders
                   for img_name in os.listdir(os.path.join(IMAGES_DIR, breed_folder))
                   if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Hata:", e)

# Modeli eğit
def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        directory=CROPPED_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        workers=16,
        max_queue_size=64
    )

    val_generator = train_datagen.flow_from_directory(
        directory=CROPPED_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        workers=16,
        max_queue_size=64
    )

    num_classes = len(train_generator.class_indices)
    print(f"Toplam sınıf sayısı: {num_classes}")

    if os.path.exists(MODEL_SAVE_PATH):
        model = load_model(MODEL_SAVE_PATH)
    else:
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_format='tf'
    )

    periodic_checkpoint = ModelCheckpoint(
        filepath=PERIODIC_MODEL_PATH,
        save_freq=5 * len(train_generator),  # Her 5 epoch'ta bir kaydet
        verbose=1,
        save_format='tf'
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr, periodic_checkpoint]
    )

# İşlemleri çalıştır
create_cropped_dataset_multithread()
train_model()
print("Tüm işlem tamamlandı!")
