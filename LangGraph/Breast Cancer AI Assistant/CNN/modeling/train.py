# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.callbacks import LearningRateScheduler
# from config import CHECKPOINT_PATH, MODEL_SAVE_PATH
# from tensorflow.keras.optimizers import Adam

# def lr_schedule(epoch):
#     initial_lr = 1e-4
#     if epoch > 10:
#         return initial_lr * 0.1
#     elif epoch > 5:
#         return initial_lr * 0.5
#     return initial_lr

# def train_model(model, X_train, y_train, X_val, y_val):
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])

#     checkpoint = ModelCheckpoint(
#         filepath=CHECKPOINT_PATH,
#         save_best_only=True,
#         save_weights_only=True,
#         monitor='val_accuracy',
#         mode='max',
#         verbose=1
#     )

#     early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#     history = model.fit(
#         X_train, y_train,
#         epochs=50,
#         batch_size=20,
#         validation_data=(X_val, y_val),
#         callbacks=[checkpoint, early_stop, LearningRateScheduler(lr_schedule)]
#     )

#     # ✅ Save the full model (architecture + weights)
#     model.save(MODEL_SAVE_PATH)
#     print(f"Model saved to {MODEL_SAVE_PATH}")

#     return history


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from config import CHECKPOINT_PATH, MODEL_SAVE_PATH
from tensorflow.keras.optimizers import Adam

def lr_schedule(epoch):
    initial_lr = 1e-4
    if epoch > 10:
        return initial_lr * 0.1
    elif epoch > 5:
        return initial_lr * 0.5
    return initial_lr

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # ✅ Data Augmentation for Training Only
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow(X_train, y_train, batch_size=20)

    checkpoint = ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 20,  # Adjust based on batch size
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[checkpoint, early_stop, LearningRateScheduler(lr_schedule)]
    )

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return history
