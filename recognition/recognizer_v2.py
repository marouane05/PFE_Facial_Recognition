from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

#facetracker = load_model('Master2022_FaceDetector_v4.h5')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
facetracker = load_model('../facetracker_v3.h5')

id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['none','Marouane', 'Obama','ismail ','karima']

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]
    #rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faceCascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    minW = 0.1 * cap.get(3)
    minH = 0.1 * cap.get(4)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        resized = tf.image.resize(rgb, (120, 120))

        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]
        print("coord",yhat)

        if yhat[0] > 0.5:
            # Controls the main rectangle
            #les coordonnées
            """x1 = np.multiply(sample_coords[1], 450).astype(int)
            x2 = np.multiply(sample_coords[0], 450).astype(int)
            y2 = np.multiply(sample_coords[2], 450).astype(int)
            y1 = np.multiply(sample_coords[3], 450).astype(int)
            h1= x2-x1
            h2=y2-y1

            print("confidence",confidence)


            print("sample",np.multiply(sample_coords[2:],450).astype(int))




            """
            """x1= print("x",np.multiply(sample_coords[2], 450).astype(int))
            y1=print("x", np.multiply(sample_coords[3], 450).astype(int))
            x2=print("x", np.multiply(sample_coords[2], 450).astype(int))
            y2=print("x", np.multiply(sample_coords[3], 450).astype(int))
            """
            """

            #print("y",tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)))

            """
            cv2.rectangle(frame,
                          tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                          (255, 0, 0), 2)
            # Controls the label rectangle
            cv2.rectangle(frame,
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [80, 0])),
                          (255, 0, 0), -1)

            # Controls the text rendered
            cv2.putText(frame, str(confidence+""+id), tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                    [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('EyeTrack', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()