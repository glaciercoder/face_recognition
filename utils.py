import dlib
import numpy as np
import os
import glob
from imutils import face_utils
from imutils import resize
import cv2
import time
from facealigner import FaceAligner
trainingset_path = "dataset/training_set"

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
aligner = FaceAligner(shape_predictor, desiredFaceWidth=256)
win = dlib.image_window()

for nameclasses in os.listdir(trainingset_path):
    single_set_name = os.path.join(trainingset_path, nameclasses)
    for f in glob.glob(os.path.join(single_set_name, "*.jpg")):
        max_square = 0
        max_id = 0
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        win.clear_overlay()
        win.set_image(img)

        dets = face_detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Choose the main face
            square = (d.right() - d.left()) * (d.top() - d.bottom())
            if square > max_square:
                max_square = square
                max_id = k
        print("Main face is {}".format(max_id))
        d = dets[max_id]
        # Get the landmarks/parts for the face in box d.
        shape = shape_predictor(img, d)
        # Convert landmark to np array
        landmark = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(d)
        faceOrig = resize(img[y:y + h, x:x + w], width=256)
        faceAligned = aligner.align(img, landmark, d)
        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)
        
        print("Computing descriptor on aligned image ..")
        face_chip = dlib.get_face_chip(img, shape)
        dlib.hit_enter_to_continue()











