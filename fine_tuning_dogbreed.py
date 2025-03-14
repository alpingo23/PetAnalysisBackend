import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import datetime
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# Mixed precision'ı etkinleştir
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# -------------------------------------------------------------------------
# KULLANICI AYARLARI - Dizinler ve benzeri
# -------------------------------------------------------------------------
IMAGES_DIR = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\images'
ANNOTATIONS_DIR = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\annotations'
CROPPED_DIR = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\cropped_images'
MODEL_SAVE_PATH = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\stanford_dogs_model.h5'
CHECKPOINT_JSON = r'C:\Users\alpip\face-analysis-app\dog-breed-backend\training_checkpoint.json'

PHASE1_EPOCHS = 40
PHASE2_EPOCHS = 40

# -------------------------------------------------------------------------
# 1) SimpleLogger Callback - ETA, epoch süresi, vs.
# -------------------------------------------------------------------------
class SimpleLogger(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()  
        self.epoch_times = []
        self.total_epochs = self.params['epochs']

        start_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[SimpleLogger] Training started at {start_str}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        this_epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(this_epoch_time)

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)

        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)

        print(
            f"[SimpleLogger] Epoch {epoch+1}/{self.total_epochs} ended.\n"
            f"  - loss: {loss:.4f}, accuracy: {acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}\n"
            f"  - This epoch time: {this_epoch_time:.1f}s, avg epoch time: {avg_epoch_time:.1f}s\n"
            f"  - ETA: {eta_seconds/60:.1f} min (finishes around {finish_time.strftime('%Y-%m-%d %H:%M:%S')})"
        )

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        total_minutes = total_time / 60
        end_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[SimpleLogger] Training ended at {end_str}, total time: {total_minutes:.1f} min.")

# -------------------------------------------------------------------------
# 2) Dataset Kırpma Aşaması (Multithread)
# -------------------------------------------------------------------------
def process_image(breed_folder, img_name):
    breed_path = os.path.join(IMAGES_DIR, breed_folder)
    annotation_breed_path = os.path.join(ANNOTATIONS_DIR, breed_folder)
    cropped_breed_path = os.path.join(CROPPED_DIR, breed_folder)

    os.makedirs(cropped_breed_path, exist_ok=True)

    img_path = os.path.join(breed_path, img_name)
    base_name = os.path.splitext(img_name)[0]
    annotation_file = os.path.join(annotation_breed_path, base_name)

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
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    if xmax <= xmin or ymax <= ymin:
        return

    cropped_img = img_cv[ymin:ymax, xmin:xmax]
    if cropped_img.size == 0 or cropped_img.shape[0] < 10 or cropped_img.shape[1] < 10:
        print(f"Uygunsuz boyut: {img_name}, atılıyor...")
        return

    save_path = os.path.join(cropped_breed_path, img_name)
    cv2.imwrite(save_path, cropped_img)

def create_cropped_dataset_multithread():
    os.makedirs(CROPPED_DIR, exist_ok=True)
    breed_folders = [d for d in os.listdir(IMAGES_DIR)
                     if os.path.isdir(os.path.join(IMAGES_DIR, d))]

    with ThreadPoolExecutor(max_workers=15) as executor:
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

# -------------------------------------------------------------------------
# 3) Data Generators Oluşturma (Augmentation)
# -------------------------------------------------------------------------
def get_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        CROPPED_DIR,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        subset='training'
    )
    val_generator = val_datagen.flow_from_directory(
        CROPPED_DIR,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, val_generator

# -------------------------------------------------------------------------
# 4) Eğitim Durumunu Kaydetme ve Yükleme
# -------------------------------------------------------------------------
def save_training_state(epoch, model, phase):
    state = {
        'epoch': epoch,
        'phase': phase,
        'model_path': MODEL_SAVE_PATH
    }
    with open(CHECKPOINT_JSON, 'w') as f:
        json.dump(state, f)

def load_training_state():
    if os.path.exists(CHECKPOINT_JSON):
        with open(CHECKPOINT_JSON, 'r') as f:
            return json.load(f)
    return {'epoch': 0, 'phase': 1}

# -------------------------------------------------------------------------
# 5) Phase 1 ve Phase 2 Eğitim Fonksiyonları
# -------------------------------------------------------------------------
def phase_1(train_generator, val_generator, model, start_epoch=0):
    print(f"\n=== [PHASE 1] Fine-tuning block_15 & block_16 (starting from epoch {start_epoch + 1}) ===")
    unfreeze = False
    for layer in reversed(model.layers):
        if 'block_15' in layer.name or 'block_16' in layer.name:
            unfreeze = True
        if unfreeze:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    logger = SimpleLogger()
    checkpoint_callback = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    for epoch in range(start_epoch, PHASE1_EPOCHS):
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=1,
            initial_epoch=epoch,
            callbacks=[checkpoint_callback, early_stop, logger, reduce_lr]
        )
        save_training_state(epoch + 1, model, 1)
        if early_stop.stopped_epoch > 0:
            break
    print("=== Phase 1 tamamlandı ===")
    return model

def phase_2(train_generator, val_generator, model, start_epoch=0):
    print(f"\n=== [PHASE 2] Fine-tuning block_12, block_13, block_14, block_15, block_16 (starting from epoch {start_epoch + 1}) ===")
    unfreeze = False
    for layer in reversed(model.layers):
        if 'block_12' in layer.name:
            unfreeze = True
        if unfreeze:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    logger = SimpleLogger()
    checkpoint_callback = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    for epoch in range(start_epoch, PHASE2_EPOCHS):
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=1,
            initial_epoch=epoch,
            callbacks=[checkpoint_callback, early_stop, logger, reduce_lr]
        )
        save_training_state(epoch + 1, model, 2)
        if early_stop.stopped_epoch > 0:
            break
    print("=== Phase 2 tamamlandı ===")
    return model

# -------------------------------------------------------------------------
# 6) Model Değerlendirme Fonksiyonları
# -------------------------------------------------------------------------
def evaluate_model(model, test_generator):
    print("\nModel test veri seti üzerinde değerlendiriliyor...")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def detailed_evaluation(model, test_generator):
    print("\nDetaylı değerlendirme yapılıyor...")
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)

# -------------------------------------------------------------------------
# 7) Main Training Fonksiyonu
# -------------------------------------------------------------------------
def two_phase_training():
    create_cropped_dataset_multithread()
    train_generator, val_generator = get_data_generators()
    num_classes = len(train_generator.class_indices)
    print("class_indices:", train_generator.class_indices)
    print("Toplam sınıf sayısı:", num_classes)

    # Test generator (doğrulama setini kullanıyoruz, ayrı bir test seti önerilir)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        CROPPED_DIR,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Modeli yükle veya oluştur
    if os.path.exists(MODEL_SAVE_PATH):
        print("Önceden kaydedilmiş model bulundu, yüklüyor...")
        model = load_model(MODEL_SAVE_PATH)
    else:
        print("Kaydedilmiş model yok, sıfırdan oluşturuluyor...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Eğitim durumunu yükle
    state = load_training_state()
    start_epoch = state['epoch']
    start_phase = state['phase']

    if start_phase == 1:
        print("\n>>>>>>>> Phase 1 başlıyor...\n")
        model = phase_1(train_generator, val_generator, model, start_epoch)
        start_epoch = 0  # Phase 2 için sıfırla

    if start_phase <= 2:
        print("\n>>>>>>>> Phase 2 başlıyor...\n")
        model = phase_2(train_generator, val_generator, model, start_epoch)

    print("\nTüm fazlar tamamlandı. Model kaydedildi/güncellendi.\n")

    # Modeli değerlendir
    evaluate_model(model, test_generator)
    detailed_evaluation(model, test_generator)

# -------------------------------------------------------------------------
# Çalıştır
# -------------------------------------------------------------------------
if __name__ == "__main__":
    two_phase_training()
    print("Tüm işlem bitti. Koddur koltur, ver elini uyku!")