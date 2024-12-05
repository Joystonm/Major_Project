# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage
# from django.conf import settings
# import os
# import cv2
# import numpy as np
# from keras.models import Model
# from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, Conv2DTranspose
# from keras.optimizers import Adam
# from keras import backend as keras

# # Define Dice Coefficient and Loss
# def dice_coef(y_true, y_pred):
#     y_true_f = keras.flatten(y_true)
#     y_pred_f = keras.flatten(y_pred)
#     intersection = keras.sum(y_true_f * y_pred_f)
#     return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

# # Define the CNN model
# def getCNNModel(input_size=(128,128,1)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv1)
#     conv1 = Dropout(0.1)(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv2)
#     conv2 = Dropout(0.1)(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Conv2D(512, (3, 3), dilation_rate=2, activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

#     up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#     conv6 = Conv2D(256, (3, 3), dilation_rate=2, activation='relu', padding='same')(up6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

#     up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#     conv7 = Conv2D(128, (3, 3), dilation_rate=2, activation='relu', padding='same')(up7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

#     up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#     conv8 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', padding='same')(up8)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
#     conv9 = Conv2D(32, (3, 3), dilation_rate=2, activation='relu', padding='same')(up9)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

#     conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
#     return Model(inputs=[inputs], outputs=[conv10])

# # Prediction function
# def predict(image_path, cnn_model):
#     img = cv2.imread(image_path, 0)  # Use the direct path for image
#     img_resized = cv2.resize(img, (128, 128))
#     img_normalized = (img_resized - 127.0) / 127.0
#     img_input = img_normalized.reshape(1, 128, 128, 1)
#     preds = cnn_model.predict(img_input)
#     preds = (preds[0] * 255).astype(np.uint8)
    
#     # Save to media directory
#     segmented_filename = os.path.join(settings.MEDIA_ROOT, 'segmented_' + os.path.basename(image_path))
#     cv2.imwrite(segmented_filename, preds)

#     # Calculate the percentage of white pixels in the segmented mask
#     white_pixel_percentage = np.mean(preds) * 100

#     if white_pixel_percentage < 1:
#         tumor_status = 'No tumor'
#     else:
#         tumor_status = 'Tumor present'

#     return os.path.join(settings.MEDIA_URL, 'segmented_' + os.path.basename(image_path)), tumor_status



# # Views
# def home(request):
#     return render(request, 'home.html')

# def detection(request):
#     if request.method == 'POST' and request.FILES['image']:
#         image_file = request.FILES['image']
#         fs = FileSystemStorage()
#         filename = fs.save(image_file.name, image_file)
#         image_url = fs.url(filename)

#         cnn_model = getCNNModel(input_size=(128, 128, 1))
#         cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
#         weights_path = os.path.join(settings.BASE_DIR, 'Liver_Detection', 'static', 'models', 'cnn_weights.hdf5')
#         cnn_model.load_weights(weights_path)

#         # Use the actual path of the uploaded image
#         uploaded_image_path = fs.path(filename)
#         segmented_url, tumor_status = predict(uploaded_image_path, cnn_model)

#         return render(request, 'detection.html', {
#             'image_url': image_url,
#             'segmented_url': segmented_url,
#             'tumor_status': tumor_status  # Pass the tumor status to the template
#         })

#     return render(request, 'detection.html')



from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as keras
from skimage.measure import regionprops  # Import regionprops
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.segmentation import clear_border

# Views
def home(request):
    return render(request, 'home.html')

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

def predict(image_path, model):
    # Read and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Make prediction
    pred = model.predict(img)
    pred = (pred > 0.5).astype(np.uint8)
    
    # Save the segmented mask
    cv2.imwrite("segmented_mask.png", pred[0, :, :, 0] * 255)
    
    return pred[0, :, :, 0]

def classify_tumor_stage(segmented_mask):
    white_pixel_percentage = np.mean(segmented_mask) * 100
    if white_pixel_percentage < 1:
        return 'No tumor'
    elif white_pixel_percentage < 1.5:
        return 'Stage I: Early Stage'
    elif white_pixel_percentage < 5:
        return 'Stage II: Intermediate Stage'
    elif white_pixel_percentage < 10:
        return 'Stage III: Advanced Stage'
    else:
        return 'Stage IV: Critical Stage'

# def highlight_tumor(original_image, mask):
#     mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
#     highlighted_image = original_image.copy()
#     red_overlay = np.zeros_like(original_image)
#     red_overlay[mask > 0] = [0, 0, 255]  # Red color in BGR
#     alpha = 0.3
#     highlighted_image = cv2.addWeighted(highlighted_image, 1, red_overlay, alpha, 0)
#     return highlighted_image
def highlight_tumor(original_image, mask):
    mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    highlighted_image = original_image.copy()
    green_overlay = np.zeros_like(original_image)
    green_overlay[mask > 0] = [0, 255, 0]  # Green color in BGR
    alpha = 0.3
    highlighted_image = cv2.addWeighted(highlighted_image, 1, green_overlay, alpha, 0)
    return highlighted_image

# def detect_tumor(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         # Handle uploaded file
#         uploaded_file = request.FILES['image']
#         fs = FileSystemStorage(location=settings.MEDIA_ROOT)  # Use MEDIA_ROOT for file storage
#         image_path = fs.save(uploaded_file.name, uploaded_file)
        
#         # Get the absolute file path
#         absolute_image_path = os.path.join(settings.MEDIA_ROOT, image_path)

#         # Load the CNN model
#         cnn_model = getCNNModel(input_size=(128, 128, 1))
#         cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
#         weights_path = os.path.join(settings.BASE_DIR, 'Liver_Detection', 'static', 'models', 'cnn_weights.hdf5')
#         cnn_model.load_weights(weights_path)

#         # Prediction - Process the uploaded image
#         img_original = cv2.imread(absolute_image_path)
#         segmented_mask = predict(absolute_image_path , cnn_model)
#         tumor_stage = classify_tumor_stage(segmented_mask)
#         img_mask = cv2.imread("segmented_mask.png", cv2.IMREAD_GRAYSCALE)

#         # Highlight tumor
#         highlighted_image = highlight_tumor(img_original, img_mask)

#         # Tumor Size Analysis
#         tumor_size = np.sum(segmented_mask) / (segmented_mask.shape[0] * segmented_mask.shape[1])
#         sizes = ['Tumor', 'Healthy Tissue']
#         values = [tumor_size, 1 - tumor_size]

#         # Intensity Distribution
#         intensity_hist = cv2.calcHist([img_original], [0], None, [50], [0, 256])

#         # Confidence Heatmap
#         confidence_map = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

#         # Region Properties
#         props = regionprops(segmented_mask.astype(int))
#         region_metrics = []
#         metrics = []  # Initialize metrics list
#         region_values = []  # Initialize values list
        
#         if len(props) > 0:
#             metrics = ['Area', 'Perimeter', 'Eccentricity']
#             region_values = [props[0].area, props[0].perimeter, props[0].eccentricity]
#             region_metrics = list(zip(metrics, region_values))

#         # Create the bar chart for Region Properties
#         plt.figure(figsize=(6, 4))
#         plt.bar(metrics, region_values, color='purple')  # Use region_values instead of values
#         plt.title("Tumor Region Properties", fontsize=14, fontweight='bold')
#         # plt.xticks(rotation=180)

#         # Save the plot to a BytesIO object
#         buf = BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)

#         # Encode the plot as base64
#         region_properties_base64 = base64.b64encode(buf.read()).decode('utf-8')

#         # Close the plot to free memory
#         plt.close()

#         # Edge Detection
#         edges = cv2.Canny(img_mask, 100, 200)

#         # Convert images to display in HTML
#         def convert_image_to_base64(image):
#             _, buffer = cv2.imencode('.png', image)
#             return base64.b64encode(buffer).decode('utf-8')

#         original_image_base64 = convert_image_to_base64(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
#         mask_image_base64 = convert_image_to_base64(img_mask)
#         highlighted_image_base64 = convert_image_to_base64(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
#         confidence_map_base64 = convert_image_to_base64(cv2.cvtColor(confidence_map, cv2.COLOR_BGR2RGB))
#         edges_base64 = convert_image_to_base64(edges)

#         # Prepare context to pass to the template
#         context = {
#             'tumor_stage': tumor_stage,
#             'tumor_size': tumor_size * 100,
#             'sizes': sizes,
#             'values': values,
#             'intensity_hist': intensity_hist,
#             'region_metrics': region_metrics,
#             'region_properties': region_properties_base64,
#             'original_image': original_image_base64,
#             'mask_image': mask_image_base64,
#             'highlighted_image': highlighted_image_base64,
#             'confidence_map': confidence_map_base64,
#             'edges': edges_base64,
#         }

#         return render(request, 'detection.html', context)

#     else:
#         # For GET requests, render the form to upload an image
#         return render(request, 'detection.html')

def detect_tumor(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Handle uploaded file
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)  # Use MEDIA_ROOT for file storage
        image_path = fs.save(uploaded_file.name, uploaded_file)
        absolute_image_path = os.path.join(settings.MEDIA_ROOT, image_path)

        # Load the CNN model
        cnn_model = getCNNModel(input_size=(128, 128, 1))
        cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
        weights_path = os.path.join(settings.BASE_DIR, 'Liver_Detection', 'static', 'models', 'cnn_weights.hdf5')
        cnn_model.load_weights(weights_path)

        # Prediction - Process the uploaded image
        img_original = cv2.imread(absolute_image_path)
        segmented_mask = predict(absolute_image_path, cnn_model)
        tumor_stage = classify_tumor_stage(segmented_mask)
        img_mask = cv2.imread("segmented_mask.png", cv2.IMREAD_GRAYSCALE)

        # Highlight tumor
        highlighted_image = highlight_tumor(img_original, img_mask)

        # Tumor Size Analysis
        tumor_size = np.sum(segmented_mask) / (segmented_mask.shape[0] * segmented_mask.shape[1])
        sizes = ['Tumor', 'Healthy Tissue']
        values = [tumor_size, 1 - tumor_size]

        # Generate the Tumor Size Analysis Pie Chart
        plt.figure(figsize=(6, 4))
        plt.pie(values, labels=sizes, autopct='%1.1f%%', colors=['red', 'lightgray'])
        plt.title("Tumor vs Healthy Tissue Distribution", fontsize=14, fontweight='bold')

        # Save the pie chart as a base64-encoded image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        tumor_size_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Close the plot to free memory
        plt.close()

        # Other visualizations and metrics
        intensity_hist = cv2.calcHist([img_original], [0], None, [50], [0, 256])
        confidence_map = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
        
                # Initialize metrics and region_values with default values
        metrics = []
        region_values = []

        props = regionprops(segmented_mask.astype(int))
        region_metrics = []

        if len(props) > 0:
            metrics = ['Area', 'Perimeter', 'Eccentricity']
            region_values = [props[0].area, props[0].perimeter, props[0].eccentricity]
            region_metrics = list(zip(metrics, region_values))

        # Region Properties Bar Chart
        plt.figure(figsize=(6, 4))
        plt.bar(metrics, region_values, color='purple')  # Will render an empty chart if no metrics
        plt.title("Tumor Region Properties", fontsize=14, fontweight='bold')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        region_properties_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        # Edge Detection
        edges = cv2.Canny(img_mask, 100, 200)

        # Convert images to base64
        def convert_image_to_base64(image):
            _, buffer = cv2.imencode('.png', image)
            return base64.b64encode(buffer).decode('utf-8')

        original_image_base64 = convert_image_to_base64(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        mask_image_base64 = convert_image_to_base64(img_mask)
        highlighted_image_base64 = convert_image_to_base64(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
        confidence_map_base64 = convert_image_to_base64(cv2.cvtColor(confidence_map, cv2.COLOR_BGR2RGB))
        edges_base64 = convert_image_to_base64(edges)

        # Context for rendering the template
        context = {
            'tumor_stage': tumor_stage,
            'tumor_size': tumor_size * 100,
            'sizes': sizes,
            'values': values,
            'tumor_size_chart': tumor_size_chart_base64,
            'intensity_hist': intensity_hist,
            'region_metrics': region_metrics,
            'region_properties': region_properties_base64,
            'original_image': original_image_base64,
            'mask_image': mask_image_base64,
            'highlighted_image': highlighted_image_base64,
            'confidence_map': confidence_map_base64,
            'edges': edges_base64,
        }

        return render(request, 'detection.html', context)

    else:
        # Render the form for GET requests
        return render(request, 'detection.html')
