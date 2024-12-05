# **Liver Tumor Detection and Analysis using CNN**

## **About the Project**  
This project focuses on detecting and classifying liver tumors from CT scan images using deep learning techniques. The system utilizes image processing methods and a Convolutional Neural Network (CNN) model for accurate tumor segmentation and classification based on size, stage, and region. The project also integrates a Django-based web application to provide a user-friendly interface for uploading CT scan images and obtaining tumor analysis results.

### **Features**  
- **Image Preprocessing**: Thresholding, normalization, connected component labeling, and color mapping.  
- **Tumor Segmentation**: Uses a CNN model for precise tumor detection.  
- **Tumor Classification**: Classifies tumors based on size, stage, and location using segmented masks.  
- **Metrics Calculation**: Calculates tumor size, area, perimeter, and eccentricity.  
- **Web Interface**: Provides a simple interface for uploading CT scan images and viewing results.  

---

## **Setup Instructions**  
Follow these steps to set up and run the project on your local machine:

### **Prerequisites**  
#### Install Django
pip install django

####  Install Pillow for file handling
pip install Pillow

####  Install OpenCV
pip install opencv-python

####  Install Matplotlib (non-GUI backend)
pip install matplotlib

####  Install NumPy
pip install numpy

####  Install Keras
pip install keras

####  Install Scikit-image for image processing
pip install scikit-image


### **Installation and Usage**  

1. **Activate Virtual Environment**  
   ```bash  
   \Liver_env\Scripts\activate
2. **Navigate to the project directory**
      ```bash 
   cd Liver_app
3. **Run the Django development server**
      ```bash 
   python manage.py runserver
