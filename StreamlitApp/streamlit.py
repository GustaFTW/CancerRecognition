import streamlit as st
from PIL import Image
# import tensorflow as tf
# from PIL import ImageOps
# import numpy as np
# import io


# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model("C:\Projects\Project1\CancerRecognition\SavedModel\mymodel.h5")
#     return model
# model = load_model()
# class_names = ['Astrocitoma', 'Carcinoma', 'Ependimoma', 
#                'Ganglioglioma', 'Germinoma', 'Glioblastoma', 'Granuloma', 
#                'Meduloblastoma', 'Meningioma', 'Neurocitoma', 'No Can', 'Oligodendroglioma', 
#                'Papiloma', 'Schwannoma', 'Tuberculoma']
img1 = Image.open("StreamlitApp/pics/label_distribution.png")
img2 = Image.open("StreamlitApp/pics/image_classes.png")
img3 = Image.open("StreamlitApp/pics/plot_loss_curves.png")
img4 = Image.open("StreamlitApp/pics/confusion_matrix.png")
img5 = Image.open("StreamlitApp/pics/classification_report.png")

st.title("Cancer Recognition Project")
st.text("""This is a project that has the goal of creating an AI that can identify 15 
types of cancer, using a feature extraction model with some regularization layers 
to prevent overfitting. The model used a Kaggle's dataset of 44 types of cancer, 
but for this problem only 15 classes were used (dataset can be found in 
https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c). The goal 
of this AI is help with the diagnose of such a harmfull and dangerous disease.
With this AI deployed in the step after an MRI exam can help maybe catch something
that may have passed by a doctor, and cancer being a disease that should be 
diagnosed as soon as possible (check for more: https://www.cancerresearchuk.org/
about-cancer/cancer-symptoms/why-is-early-diagnosis-important), every techonology
that could help with the early diagnose is helpfull.""")
st.title("Dataset")
st.text("""As mentioned before, this project was feed to a Kaggle's dataset and the 
data had to be reduced to 15 classes instead of the original 44 (due 
to limited resources the training accuracy wouldn't reach much higher 
values with 44 different classes). Here's a distribution of the 15 
classes (No can stands for no cancer detected): """)
st.image(img1)
st.text("A preview of every class:")
st.image(img2)
st.title("Architecture of the model")
st.text("""The model was a simple model that after the layers of the efficientnetb5 
base model, it was added some regularization layers so that the model wouldn't overfit, 
the model achieved 97% validation accuracy on 13 epochs.""")
st.title("Evaluating the model")
st.text("Here are some plots for the evaluation (click to full view): ")
st.title("Plot Loss Curves")
st.test("")


st.title("Conclusion")
st.text("""With the field of AI evolving much as it is every day, the use of AI in 
several should not only be encourage as well as needed. Many trivial and everyday 
problems can be solved with AI, but it shouldn't stop there, real and complex 
problem are (and will continued to be) solved by AI, this field should be embraced 
as much as possible so it can give astonishing results!""")
# st.title("Try it yourself")
# st.text("""Try uploading a foto of a MRI exam and see what the model will classify it as
# (please notice that for any diagnose a doctor should be notified)""")
# # Use the file uploader to get a file from the user
# file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# def import_and_predict(image_data, model):
#     size = (224, 224)
#     image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#     img = np.asarray(image)
#     img_reshape = tf.expand_dims(img, axis=0)
#     prediction=model.predict(img_reshape)
#     # return tf.argmax(prediction, axis=1)
#     return class_names[tf.argmax(tf.squeeze(prediction))]


# # If the user uploaded a file, do something with it
# if file is not None:
#     try:
#         # Access the contents of the file as bytes
#         image = Image.open(file)
#         st.image(image, use_column_width=True)
#         predictions = import_and_predict(image, model)
#         st.text(predictions)
#     except Exception as e:
#         st.error("Error: Unable to process the file. Please make sure it is a valid image file.")