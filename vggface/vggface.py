from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input

# function to extract a face from a photograph
def extract_face(img_path, required_size=(224, 224)):
    
    img = plt.imread(img_path) # load the image
    detector = MTCNN() # loading the detector
    faces = detector.detect_faces(img) # detect faces in the image
    
    # getttng the coordinates of the bounding box of the image
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height
    
    face = img[y1:y2, x1:x2] # cropping the face from the image
    img = Image.fromarray(face) # converting into array
    img = img.resize(required_size) # resizing it to the required size 

    return np.asarray(img)

# function to calculate the face embeddings for a list of images
def get_embeddings(img_paths):

    faces = [extract_face(img) for img in img_paths] # extracting the faces from the provided list of images
    samples = np.asarray(faces, 'float32') # converting the faces into float arrays
    samples = preprocess_input(samples, version = 2) # centering pixels
    model = VGGFace(model = 'resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') # creating a vggface model
    yhat = model.predict(samples) # prediction
    return yhat

def compare(known_embedding, candidate_embedding, threshold = 0.4):

    dist = cosine(known_embedding, candidate_embedding) # calculate the distance between the embeddings
    if dist <= threshold:
        print('\nFace is a match')
    else:
        print('\nFace is not a match')
    print("Distance :",dist)

img_paths = ['images/img0.jpg', 'images/img1.jpg', 'images/img2.jpg', 'images/img3.jpg', 'images/img4.jpg'] # list of paths of images

embeddings = get_embeddings(img_paths) # getting the face embeddings of the list of images

anchor = embeddings[0] # image against which all other images are compared

# comparing the other images to the anchor image
compare(anchor, embeddings[1])
compare(anchor, embeddings[2])
compare(anchor, embeddings[3])
compare(anchor, embeddings[4])
