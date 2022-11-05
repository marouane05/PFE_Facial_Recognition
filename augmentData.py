import albumentations as alb

import os
import time
import uuid
import cv2
import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

#cette partie pour ajuster la taille des images (450,450)
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                       bbox_params=alb.BboxParams(format='albumentations',
                                                  label_fields=['class_labels']))

#on boucle sur tous les dossiers
for partition in ['train','test','val']:
    #on prend image par image
    for image in os.listdir(os.path.join('data2', partition, 'images')):
        #on affecte l'image au variable img
        img = cv2.imread(os.path.join('data2', partition, 'images', image))
        #on instancie les cordonnées
        coords = [0,0,0.00001,0.00001]
        #on prend le path du label fichier json
        label_path = os.path.join('data2', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            #on affecte les cordonnées
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            #on divise les cordonnées
            coords = list(np.divide(coords, [640,480,640,480]))

        try:
            for x in range(60):
                #on initialise augmentor
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                #création d'image avec les nouveaux cordonnées
                cv2.imwrite(os.path.join('augmented_data2', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation ={}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0


                with open(os.path.join('augmented_data2', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)