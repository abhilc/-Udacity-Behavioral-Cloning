import csv
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import cv2


lines = []
measurements = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

images = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

print(len(images))
print(len(measurements))

#Capture left and right camera images as well
for line in lines:
    image_left_path = line[1]
    image_right_path = line[2]
    left_fname = image_left_path.split('/')[-1]
    right_fname = image_right_path.split('/')[-1]
    path = 'data/IMG/'
    correction = 0.2
    left_img = cv2.imread(path+left_fname)
    right_img = cv2.imread(path+right_fname)
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    images.append(left_img)
    measurements.append(steering_left)
    images.append(right_img)
    measurements.append(steering_right)

lines = []

'''
with open('/opt/carnd_p3/new_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/opt/carnd_p3/new_data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    steering_center = float(line[3])
    correction = 0.1
    lpath = line[1]
    rpath = line[2]
    lname = lpath.split('/')[-1]
    rname = rpath.split('/')[-1]
    path = '/opt/carnd_p3/new_data/IMG/'
    #print(path+lname)
    limg = cv2.imread(path+lname)
    rimg = cv2.imread(path+rname)
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    images.append(limg)
    measurements.append(steering_left)
    images.append(rimg)
    measurements.append(steering_right)
'''
#print(len(images))
#print(len(measurements))
#


augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement*-1)

#print(len(augmented_images))


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

   
#model = Sequential()
#model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
#model.add(Flatten()) 
#model.add(Dense(1))

#model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

#model.save('model.h5')

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, kernel_size=5, strides=2, padding="same", activation="relu"))
model.add(Conv2D(36, kernel_size=5, strides=2, padding="same", activation="relu"))
model.add(Conv2D(48, kernel_size=5, strides=2, padding="same", activation="relu"))
model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

'''
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
model.save('model_nvidia_with_allcams_3.h5')


