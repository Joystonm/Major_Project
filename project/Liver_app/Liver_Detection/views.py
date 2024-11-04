from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as keras

# Define Dice Coefficient and Loss
def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Define the CNN model
def getCNNModel(input_size=(128,128,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv1)
    conv1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv2)
    conv2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    return Model(inputs=[inputs], outputs=[conv10])

# Prediction function
def predict(image_path, cnn_model):
    img = cv2.imread(image_path, 0)  # Use the direct path for image
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = (img_resized - 127.0) / 127.0
    img_input = img_normalized.reshape(1, 128, 128, 1)
    preds = cnn_model.predict(img_input)
    preds = (preds[0] * 255).astype(np.uint8)
    
    # Save to media directory
    segmented_filename = os.path.join(settings.MEDIA_ROOT, 'segmented_' + os.path.basename(image_path))
    cv2.imwrite(segmented_filename, preds)

    # Calculate the percentage of white pixels in the segmented mask
    white_pixel_percentage = np.mean(preds) * 100

    if white_pixel_percentage < 1:
        tumor_status = 'No tumor'
    else:
        tumor_status = 'Tumor present'

    return os.path.join(settings.MEDIA_URL, 'segmented_' + os.path.basename(image_path)), tumor_status



# Views
def home(request):
    return render(request, 'home.html')

def detection(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_url = fs.url(filename)

        cnn_model = getCNNModel(input_size=(128, 128, 1))
        cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
        weights_path = os.path.join(settings.BASE_DIR, 'Liver_Detection', 'static', 'models', 'cnn_weights.hdf5')
        cnn_model.load_weights(weights_path)

        # Use the actual path of the uploaded image
        uploaded_image_path = fs.path(filename)
        segmented_url, tumor_status = predict(uploaded_image_path, cnn_model)

        return render(request, 'detection.html', {
            'image_url': image_url,
            'segmented_url': segmented_url,
            'tumor_status': tumor_status  # Pass the tumor status to the template
        })

    return render(request, 'detection.html')
