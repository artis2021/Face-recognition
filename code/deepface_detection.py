# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:53:07 2022

@author: arti
"""

from deepface import DeepFace
import cv2

#detector backend="opencv","ssd","dlib" ,"mtcnn","retinaface"

#face detection and face alignment

face_detected=DeepFace.detectFace(img_path='testing/modi1.jpg',detector_backend="opencv")

face_detected=cv2.cvtColor(face_detected, cv2.COLOR_BGR2RGB)
cv2.imshow("face_detected", face_detected)

