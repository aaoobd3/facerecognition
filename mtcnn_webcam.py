# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:40:04 2021

@author: alaaldin_obeid
"""

from random import choice
from numpy import load
from numpy import expand_dims,asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import joblib

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=160)
model_facenet = load_model('facenet_keras.h5')
print('Loaded Model')
video_capture = cv2.VideoCapture(0)

# load faces
data = load('images.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('images-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model

filename = 'finalized_model.sav'
model = joblib.load(filename)

# test model on a random example from the test dataset
while(1):
    ret, frame = video_capture.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = detector(frame)
    for face in faces:
        
        (x, y, w, h) = face_utils.rect_to_bb(face)
        

        # Draw a label with a name below the face
        
        font = cv2.FONT_HERSHEY_DUPLEX
        
        face_img = frame_gray[y-50:y + h+100, x-50:x + w+100]
        face_aligned = face_aligner.align(frame, frame_gray, face)
        img = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        image = image.resize((160,160))
        face_pixels = asarray(image)
        face_pixels = face_pixels.astype('float32')
        	# standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        	# transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        samples = model_facenet.predict(samples)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        if class_probability>70:
            cv2.putText(frame, '{} ({})'.format(predict_names[0],class_probability), (x + 6, y - 6), font, 1.0, (255, 0, 255), 1)
        else:
            cv2.putText(frame, '{} ({})'.format('unknown',class_probability), (x + 6, y - 6), font, 1.0, (255, 0, 255), 1)
            
    cv2.imshow('Video', frame)
    cv2.imshow('face', face_aligned)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
# plot for fun
video_capture.release()
cv2.destroyAllWindows()

