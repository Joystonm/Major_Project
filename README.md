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
### **System Workflow**  
1. **Upload CT Scan Image**: The system accepts CT scan images uploaded through the web interface.  
2. **Image Preprocessing**: Techniques such as normalization and color mapping prepare the image for analysis.  
3. **Tumor Detection**: A trained CNN model identifies and segments the tumor regions.  
4. **Tumor Classification**: Tumors are classified into stages and regions based on size and other features.  
5. **Metrics Display**: Outputs include tumor size, stage, area, and other key metrics.  
6. **Results Visualization**: Segmentation maps and analytical results are displayed on the web interface.

---
### **Hardware Requirements**    

| Component           | Specification                      |
|---------------------|------------------------------------|
| **CPU**             | 8th Generation Intel® Core™ i5 Processor |
| **RAM**             | 8 GB or above                     |
| **Hard Disk**       | 500 GB or above                   |
| **GPU**             | NVIDIA GTX 1650                   |

---

### **Software Requirements**  

| Component                    | Specification                  |
|------------------------------|--------------------------------|
| **Operating System**         | Windows 11                    |
| **Programming Language**     | Python                        |
| **Backend Libraries**        | Keras, OpenCV, NumPy          |
| **Front-end Framework**      | Django                        |
| **Integrated Development Environment** | Jupyter Notebook       |

---
### **Technologies Used**

- **Programming Language**: Python  
- **Web Framework**: Django  
- **Deep Learning Framework**: TensorFlow, Keras  
- **Image Processing**: OpenCV, NumPy  
- **Visualization Tools**: Matplotlib, Seaborn  
- **Version Control**: Git, GitHub  
- **Deployment Environment**: Virtual Environment (venv)  
- **Operating System**: Windows/Linux  
---

### **Installation and Usage**  

1. **Activate Virtual Environment**  
   ``` 
   .\Liver_env\Scripts\activate
      ``` 
2. **Navigate to the project directory**
      ```
   cd Liver_app
   ``` 
3. **Run the Django development server**
      ```
   python manage.py runserver
