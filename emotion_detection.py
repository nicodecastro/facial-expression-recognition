# Import packages
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, Rescaling
from keras.optimizers import Adam 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers.schedules import ExponentialDecay 
import matplotlib.pyplot as plt 

# Data Preprocessing
train_data = image_dataset_from_directory(
    'data/train/',
    batch_size=64,
    image_size=(48, 48),
    color_mode='grayscale',
    label_mode='categorical'
)

validation_data = image_dataset_from_directory(
    'data/test/',
    batch_size=64,
    image_size=(48, 48),
    color_mode='grayscale',
    label_mode='categorical'
)

normalization_layer = Rescaling(1./255)

normalized_train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
normalized_validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))


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
    normalized_train_data, 
    epochs=50, 
    validation_data=normalized_validation_data, 
)

# Accuracy and Loss Evaluation
emotion_model.evaluate(normalized_validation_data)

# Plots for Accuracy and Loss
accuracy = emotion_model_info.history['accuracy'] 
val_accuracy = emotion_model_info.history['val_accuracy'] 
loss = emotion_model_info.history['loss'] 
val_loss = emotion_model_info.history['val_loss']
  
# Accuracy graph 
plt.subplot(1, 2, 1) 
plt.plot(accuracy, label='accuracy') 
plt.plot(val_accuracy, label='val accuracy') 
plt.title('Accuracy Graph') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
  
# loss graph 
plt.subplot(1, 2, 2) 
plt.plot(loss, label='loss') 
plt.plot(val_loss, label='val loss') 
plt.title('Loss Graph') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 
  
plt.show() 

# Save model structure in json file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)


# Save trained model weight in .h5 file
emotion_model.save_weights("emotion_model.weights.h5")