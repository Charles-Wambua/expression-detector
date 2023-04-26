from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

training_directory = 'data/train'
validation_directory = 'data/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    training_directory,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_directory,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# CNN architecture
my_model = keras.models.Sequential()

my_model.add(Conv2D(64, kernel_size=(3, 3),
             activation='relu', input_shape=(48, 48, 1)))
my_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Dropout(0.25))

my_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
my_model.add(MaxPooling2D(pool_size=(2, 2)))
my_model.add(Dropout(0.25))

my_model.add(Flatten())
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(7, activation='softmax'))

my_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the model
history = my_model.fit_generator(
    train_generator,
    steps_per_epoch=28709//64,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=7178//64)

my_model.save_weights('model_1.h5')






















































