from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
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
    batch_size=128,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_directory,
    target_size=(48, 48),
    batch_size=128,
    color_mode="grayscale",
    class_mode='categorical')

# CNN architecture
my_model = Sequential()

my_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(48,48,1)))
my_model.add(BatchNormalization())
my_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.25))

my_model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.25))

my_model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))
my_model.add(BatchNormalization())
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.25))

my_model.add(Flatten())
my_model.add(Dense(512, activation='relu'))
my_model.add(BatchNormalization())
my_model.add(Dropout(0.5))
my_model.add(Dense(7, activation='softmax'))


my_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

# Train the model
history = my_model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator))


my_model.save_weights('model.h6')























































