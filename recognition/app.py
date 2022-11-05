import tkinter as tk
from recognizer_Detection_FROM_Scratch import OpenCamera
from tkinter import Button
from tkinter import Label
from tkinter import Entry
from tkinter import Frame
import subprocess
import sys
import os
window=tk.Tk()
path = os.getcwd();
def OpenFaceRecognition():
    pa=os.path.join(path,"recognition","recognizer_Detection_FROM_Scratch.py")
    subprocess.call(["python", "recognizer_Detection_FROM_Scratch.py"])
    #os.system(pa)


frame1 = Frame(window,bg="black",width=500,height=300)
frame1.pack()

btn=Button(window, text="Ouvrir le système de reconnaissance", fg='blue',command=OpenCamera)
btn.place(x=80, y=100)
lbl=Label(window, text="This is Label widget", fg='red', font=("Helvetica", 16))
lbl.place(x=60, y=50)
txtfld=Entry(window, text="This is Entry Widget", bd=5)
txtfld.place(x=80, y=150)
window.title('Application de reconnaissance faciale- Master Big Data 2022')
window.geometry("1200x900")
window.mainloop()

