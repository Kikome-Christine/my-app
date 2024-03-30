import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(" trained_medical_plant_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element




# Sidebar
st.sidebar.title("Medical Plant Species Identification")

# Main Page
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Species Identification"])

if app_mode == "Home":
    st.header("Identifying Medical Plant Species ")
    st.image("/content/neem-tree-2-scaled.jpeg", use_column_width=True)
    st.markdown("""
        Welcome to the Medical Plant Species Identification System! 🌿🔍

        Our mission is to assist in identifying medical plant species accurately. Upload an image of a plant,
        and our system will analyze it to determine the species. Let's contribute to medical research and
        conservation efforts through plant species identification!

        ### How It Works
        1. **Upload Image:** Go to the **Species Identification** page and upload an image of a medical plant.
        2. **Analysis:** Our system will process the image using advanced computer vision algorithms to identify the species.
        3. **Results:** View the identified species along with relevant information.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for precise plant species identification.
        - **User-Friendly:** Simple and intuitive interface for easy plant species identification.
        - **Fast and Efficient:** Receive results quickly, facilitating medical research and conservation efforts.

        ### Get Started
        Click on the **Species Identification** page in the sidebar to upload an image and experience the power of our Medical Plant Species Identification System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif app_mode == "About":
    st.header("About")

    # About the Project
    st.subheader("About the Project")
    st.markdown("""
    The Medical Plant Species Identification project aims to leverage computer vision technology to
    accurately identify various medical plant species. By utilizing state-of-the-art machine learning
    algorithms, we provide a platform for researchers, botanists, and enthusiasts to quickly and
    accurately identify medical plant species, contributing to medical research and conservation efforts.
    """)

    # Project Features
    st.subheader("Project Features")
    st.markdown("""
    - **Dataset:** The project utilizes a comprehensive dataset containing images of various medical plant species.
    - **Model:** We employ a deep learning model trained on the dataset to achieve accurate plant species identification.
    - **Application:** The project provides a user-friendly web application for easy plant species identification.
    """)

    # Team Members
    st.subheader("Team Members")

    # Define team members with their names and image paths
    team_members = {
        'OKUMU GEOFFERY (Project Lead)': '/content/OKUMU.jpg',
        'KIKOME CHRISTINE (Machine Learning Engineer)': '/content/1.jpg',
        'WAGISHA EMMANUEL (Web Developer)': '/content/WAGISHA.jpg'
    }

    # Display team members' names and images horizontally with circular images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('/content/OKUMU.jpg', width=100, use_column_width=False, caption='OKUMU GEOFFERY (Project Lead)', output_format='PNG')

    with col2:
        st.image('/content/1.jpg', width=100, use_column_width=False, caption='KIKOME CHRISTINE (Machine Learning Engineer)', output_format='PNG')

    with col3:
        st.image('/content/WAGISHA.jpg', width=100, use_column_width=False, caption='WAGISHA EMMANUEL (Web Developer)', output_format='PNG')

#Prediction Page
elif app_mode == "Species Identification":
    st.header("Species Identification")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
      st.image(test_image,width=4,use_column_width=True)
    if(st.button("Predict")):
      st.snow()
      st.write("Our Prediction")
      result_index = model_prediction(test_image)
      class_name = {'Bohera': 'Terminalia bellirica',
                    'Devilbackbone': 'Euphorbia tithymaloides',
                    'Haritoki': 'Terminalia chebula',
                    'Lemongrass': 'Cymbopogon citratus',
                    'Nayontara': 'Catharanthus roseus',
                    'Neem': 'Azadirachta indica',
                    'Pathorkuchi': 'Kalanchoe pinnata',
                    'Thankuni': 'Centella asiatica',
                    'Tulsi': 'Ocimum tenuiflorum',
                    'Zenora': ''}

      st.success("Model is Predicting it's a {}".format(class_name[result_index]))
