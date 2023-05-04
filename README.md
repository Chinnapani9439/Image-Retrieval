# **Image-Retrieval**
## **Image Retrieval with Deep Learning using Streamlit app**

### **Introduction**
This code is a Streamlit application for an Image-Based Image Retrieval model. The model retrieves the most similar images to the input image by computing the Euclidean distance between their embeddings.

The application allows the user to upload an image, which is saved locally and on Google Cloud Storage (GCS). The saved image is used as input for the retrieval model. Additionally, the user can choose an image from a list of images saved in a GCS bucket to compare to the uploaded image.

### **The application was built using the following technologies:**

- Streamlit - for building the web application
- PIL - for opening and manipulating images
- Google Cloud Storage (GCS) - for storing and retrieving images
- gcsfs - for interfacing with GCS
- NumPy - for working with arrays and numerical data
- Pandas - for data manipulation and analysis
- OpenCV - for image processing
- Matplotlib - for visualizing data
- TensorFlow - for loading the image retrieval model

## **Getting Started**

### **Prerequisites**

- Python 3.7 or later
- A Google Cloud account with a project set up
- A GCS bucket to store and retrieve images
- A TensorFlow model for image retrieval

### **Installation**

- Clone this repository: https://github.com/Chinnapani9439/Image-Retrieval.git
- Install the required packages: pip install streamlit pillow google-cloud-storage gcsfs numpy pandas opencv-python-headless matplotlib tensorflow
- Set up authentication for GCS by following the instructions in the Google Cloud documentation.
- Download the trained image retrieval model and place it in the root directory of the cloned repository.
- Place the dataset of images that the model was trained on in a folder called dataset in the root directory of the cloned repository.
- Create a CSV file containing the image details of the dataset. The CSV should have columns for id, number_type, product_id, and type.
- Create a CSV file containing the unique article types and their corresponding numbers.

### **Usage**

- Navigate to the root directory of the cloned repository.
- Start the Streamlit application by running: streamlit run app.py
- Upload an image or select an image from the list of images in the GCS bucket.
- The most similar images to the uploaded or selected image will be displayed.

### **Contributing**
Contributions to this project are welcome. If you find any issues or would like to suggest new features, please open an issue or a pull request.
