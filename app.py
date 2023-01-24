# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model, load_model
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow import keras

# visulizations
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import seaborn as sns
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from numpy import asarray
from numpy import genfromtxt
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras import backend as K
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import random
import time
import cv2
import glob
from scipy import misc
import pathlib
from imageio import imread
import streamlit as st
from mtcnn.mtcnn import MTCNN
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

def main():
  facenet = keras.models.load_model('facenet.h5')
  facenet.load_weights('facenet_keras_weights.h5')
  st.title("Face recognition")
  # load the face dataset
  data = np.load('5-celebrity-faces-embeddings.npz')
  trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
  in_encoder = Normalizer()
  emdTrainX_norm = in_encoder.transform(trainX)
  emdTestX_norm = in_encoder.transform(testX)
  outencoder = LabelEncoder()
  outencoder.fit(trainy)
  trainy_enc = outencoder.transform(trainy)
  testy_enc = outencoder.transform(testy)
  # fit model
  classifier = SVC(kernel='linear', probability=True)
  classifier.fit(emdTrainX_norm, trainy_enc)

  uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
  if uploaded_file is not None:
      # Use MTCNN to detect and align faces in the image
      # extract a single face from a given photograph
      def extract_face(filename, required_size=(160, 160)):
          # load image from file
          image = Image.open(filename)
          # convert to RGB, if needed
          image = image.convert('RGB')
          # convert to array
          pixels = asarray(image)
          # create the detector, using default weights
          detector = MTCNN()
          # detect faces in the image
          results = detector.detect_faces(pixels)
          # extract the bounding box from the first face
          x1, y1, width, height = results[0]['box']
          # bug fix
          x1, y1 = abs(x1), abs(y1)
          x2, y2 = x1 + width, y1 + height
          # extract the face
          face = pixels[y1:y2, x1:x2]
          # resize pixels to the model size
          image = Image.fromarray(face)
          image = image.resize(required_size)
          face_array = asarray(image)
          return face_array
      face = extract_face(uploaded_file)
      # Use FaceNet to extract facial embeddings
      def get_embedding(model, face):
          # scale pixel values
          face = face.astype('float32')
          # standardization
          mean, std = face.mean(), face.std()
          face = (face-mean)/std
          # transfer face into one sample (3 dimension to 4 dimension)
          sample = np.expand_dims(face, axis=0)
          # make prediction to get embedding
          yhat = model.predict(sample)
          return yhat[0]
      embeddings = get_embedding(facenet, face)
      reshaped_embeddings = embeddings.reshape(1, -1)
      samples = in_encoder.transform(reshaped_embeddings)
      #samples = np.expand_dims(transformed_embeddings, axis=0)
      # Use the trained model to predict the person in the image
      prediction = classifier.predict(samples)
      probability = classifier.predict_proba(samples)
      class_index = prediction[0]
      class_probability = probability[0,class_index] * 100
      if class_probability > 50:
        answer = outencoder.inverse_transform(prediction)
        all_names = outencoder.inverse_transform([0,1,2,3,4])
        st.write('Predicted: \n%s \n%s' % (all_names, probability[0]*100))
        st.write(answer[0])
        classes = ['elton_john', 'jerry_seinfeld', 'madonna', 'ben_afflek', 'mindy_kaling']
        if answer[0] in classes:
          msg = answer[0]
          st.success(msg)
          st.image(face, caption="Your input image")
      else:
        st.error("U idiot")
        st.image(face, caption="Your input image")


if __name__ == '__main__':
	main()
