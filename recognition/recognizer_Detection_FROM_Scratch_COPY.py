from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

facetracker = load_model('../facetracker_v3.h5')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
new_df = pd.read_csv('./data/persons.csv')
#names = ['none','Marouane', 'Obama','karima']

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))

    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        """x, y = tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int))
        h, w =tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int))
        """

        x, y = tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int))
        w, h = tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int))
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print("confidence", confidence)
        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 100):
            df_res = new_df.loc[new_df['id'] == int(id)];
            print(df_res.to_numpy()[0])
            arr = df_res.to_numpy()[0]
            id = arr[1] + "\n" + arr[2]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            frame,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

        #cv2.imwrite("./dataset/Maroyane.jpg", gray[y:y + h, x:x + w])


        #print("coord",sample_coords[:2])

        # Controls the label rectangle

        """
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [80, 0])),
                      (255, 0, 0), -1)
        """

        # Controls the text rendered
        """"
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        """
    cv2.imshow('Face recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()