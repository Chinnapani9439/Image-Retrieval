import streamlit as st
from PIL import Image
from google.cloud import storage
import gcsfs
import numpy as np 
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as plt
gcs = storage.Client()
from tensorflow.keras.models import load_model
import math

def header1(url): 
    st.markdown(f'<p style="color:#296d98;font-size:48px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)

def header2(url): 
    st.markdown(f'<p style="color:#7D4A95;font-size:15px;border-radius:2%;"><strong> &nbsp; &nbsp;{url}</strong></p>', unsafe_allow_html=True)
    
def calculateDistance(i1, i2):
    return math.sqrt(np.sum((i1-i2)**2))

def image_selector():
    storage_client = storage.Client()
    bucket_name='content_based_image'
    bucket = storage_client.get_bucket(bucket_name)
    prefix='images/'
    iterator = bucket.list_blobs(delimiter='/', prefix=prefix)
    response = iterator._get_next_page_response()
    data=[]
    for i in response['items']:
        z='gs://'+bucket_name+'/'+i['name']
        data.append(z)
    data=data[1:]
    return data   

def predict():
    st.title("Image-based Image Retrieval")
    
    uploaded_file = st.file_uploader("Upload Image Files",type=['jpg','jpeg','png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        path = 'content_images/{name}'.format(name=uploaded_file.name)
        image.save(path)
        gcs.get_bucket('content_based_image').blob('images/{name1}'.format(name1= uploaded_file.name)).upload_from_filename('/home/jupyter/kiran/Image_Retrieval/Image-retrieval-with-deep-learning/streamlit-app/content_images/{name}'.format(name=uploaded_file.name))
    
    file_names = image_selector()
    file_names = file_names[::-1]
    file_names.append("-")
    file_names = file_names[::-1]
    file_path = st.selectbox("Choose an image", file_names)
        
    if file_path != "-":
        model = load_model('visual_product_recommend.h5')
        
        df = pd.read_csv("/unique_articaltype_number_type.csv")
            
        Image_path="/content_images/{}".format(file_path.split("/")[-1])
        classes = df["type"]
        image = cv2.imread(Image_path)
        img_col = cv2.cvtColor(image,cv2.IMREAD_COLOR)
        resized_img = cv2.resize(image, dsize=(224, 224))
        
        img=cv2.cvtColor(resized_img,cv2.COLOR_BGR2RGB) 
                              
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img_col = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
        #st.write(img)
        
        resized_img = cv2.resize(img, dsize=(50, 50))
        x_data = np.array(resized_img).reshape(-1, 50,50,1)
        x_data=x_data/255
  
        numpy_image = x_data
    
        result=model.predict(x_data)
        
        col1,col2,col3 = st.columns([1,0.5,1])
        
        with col1:
            st.write("")
            
        with col2:
            new_img = Image.open(Image_path)
            st.image(new_img)
            
        with col3:
            st.write("")
        
        index = np.argmax(result)
        header1(classes[index])
        
        typeList=[]
        new_df = pd.read_csv("/images_details_deep.csv")
        
        for i, row in new_df.iterrows(): 
            if(row["number_type"]==index):
                print(row["id"],row["number_type"])
                typeList.append(row['id'])
        
        i=0
        X_similar=[]
        X_id_similar=[]
        X_numpy=[]
        
        for imageId in typeList:
    
            Image_path="/dataset/images/"+str(imageId)+".jpg"
            image = cv2.imread(Image_path,cv2.IMREAD_GRAYSCALE)
            try:
                resized_img = cv2.resize(image, dsize=(50,50))
            except:
                print("can't read file: ", str(imageId)+".jpg")
                
            X_similar.append(resized_img)
            X_id_similar.append(imageId)
            
        
        X_numpy = np.array(X_similar).reshape(-1, 50,50,1)
        X_numpy = X_numpy/255
        
        distance_list=[]
        
        for i in range (0, len(X_numpy)):
            distance_list.append(calculateDistance(numpy_image,X_numpy[i]))
        
        sorted_distance_list=distance_list.copy()
        sorted_distance_list.sort()

        least_ten_distance=sorted_distance_list[0:10]
        
        index_distance=[]
        
        for i in range (0, len(least_ten_distance)-1):
            if(least_ten_distance[i]!=least_ten_distance[i+1]):
                index_distance.append(distance_list.index(least_ten_distance[i]))

        index_distance=index_distance[0:5]
        
        
        Image_path = []
        ids = []
        for i in range(0,len(index_distance)):
            Image_path.append("/dataset/images/"+str(X_id_similar[index_distance[i]])+".jpg")
            ids.append(X_id_similar[index_distance[i]])
        
        
        col1,col2,col3,col4,col5,col6 = st.columns([0.5,1,1,1,1,1])
        
        with col1:
            st.write("")
        with col2:
            st.image(Image_path[0])
            header2(str(ids[0]))
        with col3:
            st.image(Image_path[1])
            header2(str(ids[1]))
        with col4:
            st.image(Image_path[2])
            header2(str(ids[2]))
        with col5:
            st.image(Image_path[3])
            header2(str(ids[3]))
        with col6:
            st.image(Image_path[4])
            header2(str(ids[4]))

if __name__ == "__main__":
    predict()