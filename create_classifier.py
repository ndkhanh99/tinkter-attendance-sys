import numpy as np
from PIL import Image
import os
import cv2
from Helper.align_dataset_mtcnn import main
from Helper.classifier import mainTrain


def regFaces():
    input_dir = 'data/student/raw'
    output_dir = 'data/student/processed'
    image_size = 160
    margin = 32
    random_order = 'random_order'
    gpu_memory_fraction = 0.25
    args = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'image_size': image_size,
        'margin': margin,
        'random_order': random_order,
        'gpu_memory_fraction': gpu_memory_fraction,
        'detect_multiple_faces': False
    }
    print(args['output_dir'])
    data = main(args)

    if data == 'ok':
        startTraining(data)

    # data = 'complete reg faces'
    return
# Method to train custom classifier to recognize face


def startTraining(data):
    os.remove('model/facemodel.pkl')
    # message = request.form['status']
    if data == 'ok':
        data_dir = 'data/student/processed'
        # test_data = 'backend/data/test/align'
        args = {
            'mode': 'TRAIN',
            'data_dir': data_dir,
            'model': 'model/20180402-114759.pb',
            'classifier_filename': 'model/facemodel.pkl',
            'use_split_dataset': 'store_true',
            'batch_size': 1000,
            'image_size': 160,
            'seed': 666,
            'min_nrof_images_per_class': 20,
            'nrof_train_images_per_class': 15}
        mainTrain(args)
        data = 'complete trained'
        return


def train_classifer(name):
    # Read all the images in custom data-set
    path = os.path.join(os.getcwd()+"/data/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

    for root, dirs, files in os.walk(path):
        pictures = files

    for pic in pictures:

        imgpath = path+pic
        img = Image.open(imgpath).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(pic.split(name)[0])
        # names[name].append(id)
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("./data/classifiers/"+name+"_classifier.xml")
