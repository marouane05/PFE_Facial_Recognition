import eel
from random import randint
import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
from recognition.recognizer_Detection_FROM_Scratch import OpenCamera
import datetime
eel.init("web")
# Exposing the random_python function to javascript
"""def to_integer(dt_time):
   strr = str(dt_time.year) + str(dt_time.month) + str(dt_time.day)+str(dt_time.hour)+str(dt_time.hour)+str(dt_time.minute)+str(dt_time.second)
   return int(strr)
"""
@eel.expose
def random_python():
    print("Random function running")
    return randint(1, 100)

@eel.expose
def openFaceRecognition():

    path = os.getcwd();

    facetracker = load_model('facetracker_v3.h5')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./recognition/trainer/trainer.yml')

    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['none', 'Marouane', 'Obama', 'karima']
    cap = cv2.VideoCapture(0)
    cap.set(4, 1080)

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

            x, y = tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int))
            w, h = tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int))
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            print("confidence", confidence)
            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 100):
                id = names[id]
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

        cv2.imshow('Face recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    """
     eel.show('system.html')
    pa = os.path.join(path, "recognition", "recognizer_Detection_FROM_Scratch.py")
    #subprocess.call(["python", "recognizer_Detection_FROM_Scratch.py"])
    subprocess.call(["python", pa])
    print("Random function running")
    return randint(1, 100)
    """

@eel.expose
def ToAddPerson():
    eel.show('addperson.html')
# Start the index.html file


@eel.expose
def OpenSystem():
    facetracker = load_model('./facetracker_v3.h5')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./recognition/trainer/trainer.yml')

    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    # names related to ids: example ==> Marcelo: id=1,  etc
    new_df = pd.read_csv('./recognition/data/persons.csv')
    # names = ['none','Marouane', 'Obama','karima']

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


        cv2.imshow('Face recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


@eel.expose
def TakePictures(nom,infos):

    import pandas as pd
    import uuid

    face_cascade = cv2.CascadeClassifier('./recognition/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./recognition/data/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('./recognition/data/haarcascade_smile.xml')

    # define a video capture object

    #face_id = uuid.uuid1().int
    face_id = randint(1, 10000)
    #rand2 = to_integer(datetime.datetime.now())


    dictP = {
        'id': [face_id],
        'Name': [nom],
        'Infos': [infos]
    }

    count = 0
    vid = cv2.VideoCapture(0)

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Added Lines
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # increment counter

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)

            # print("voila", count)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

            smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 25)

            for (xs, ys, ws, sh) in smile:
                cv2.rectangle(roi_color, (xs, ys), (xs + ws, ys + sh), (255, 0, 0), 1)
            count += 1;
            cv2.imwrite("./recognition/dataset/User." + str(face_id) + '.' +
                        str(count) + ".jpg", gray[y:y + h, x:x + w])
        # Finished added lines

        # Display the resulting frame
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        cv2.imshow('Face recognition', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        """ if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
        """
        if k == 27:
            break;

        elif count >= 30:  # Take 30 face sample and stop video
            # Make data frame of above data
            df = pd.DataFrame(dictP)

            # append data frame to CSV file
            df.to_csv('./recognition/data/persons.csv', mode='a', header=False, index=False)

            break;

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('recognition/trainer/trainer.yml')
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

path = 'recognition/dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()
detector = cv2.CascadeClassifier('./recognition/data/haarcascade_frontalface_default.xml');
# function to get the images and label data

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids


eel.start("index.html")


