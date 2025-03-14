train_generator = train_datagen.flow_from_directory(
    directory=CROPPED_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
