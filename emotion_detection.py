# Import packages
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.optimizers import Adam 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers.schedules import ExponentialDecay 

# Data Preprocessing
train_data_gen = ImageDataGenerator(rescale=1./255) 
validation_data_gen = ImageDataGenerator(rescale=1./255) 


# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)


# Create model structure
emotion_model = Sequential()

emotion_model.add(Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.summary()

initial_learning_rate = 0.0001

lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96) 

optimizer = Adam(learning_rate=lr_schedule) 

emotion_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# Train the neutral network/model
emotion_model_info = emotion_model.fit( 
    train_generator, 
    steps_per_epoch=28709 // 64, 
    epochs=30, 
    validation_data=validation_generator, 
    validation_steps=7178 // 64
)

emotion_model.evaluate(validation_generator)

# Save model structure in json file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)


# Save trained model weight in .h5 file
emotion_model.save_weights("emotion_model.weights.h5")