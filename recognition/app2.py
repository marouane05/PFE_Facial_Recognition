import tkinter as tk
import tkinter.font as tkFont
from tkinter import PhotoImage
from tkinter import Label

from PIL import ImageTk, Image

class App:
    def __init__(self, root):
        #setting title
        root.title("Système de reconnaissance faciale")
        imglogo = ImageTk.PhotoImage(Image.open("logo_ibn_tofail.png"))
        #setting window size
        width=1280
        height=780
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        #bg = PhotoImage(file="logo_ibn_tofail.png")

        GButton_125=tk.Button(root)
        GButton_125["activeforeground"] = "#ffb800"
        GButton_125["bg"] = "#1e9fff"
        ft = tkFont.Font(family='Times',size=18)
        GButton_125["font"] = ft
        GButton_125["fg"] = "#f7f1f1"
        GButton_125["justify"] = "center"
        GButton_125["text"] = "Button"
        GButton_125.place(x=830,y=350,width=292,height=49)
        GButton_125["command"] = self.GButton_125_command

        GMessage_401=tk.Message(root)
        GMessage_401["anchor"] = "center"
        GMessage_401["bg"] = "#f6fafc"
        GMessage_401["cursor"] = "watch"
        ft = tkFont.Font(family='Times',size=18)
        GMessage_401["font"] = ft
        GMessage_401["fg"] = "#393d49"
        GMessage_401["justify"] = "center"
        GMessage_401["text"] = "Bienvenue chez Le système de reconnaissance Faciale - Master Big Data and Cloud  2020 - 2022 (Faculté des sciences IBN tofail)"
        GMessage_401["relief"] = "groove"
        GMessage_401.place(x=100,y=40,width=1091,height=215)

        GButton_360=tk.Button(root)
        GButton_360["bg"] = "#fe0000"
        ft = tkFont.Font(family='Times',size=18)
        GButton_360["font"] = ft
        GButton_360["fg"] = "#f7f1f1"
        GButton_360["justify"] = "center"
        GButton_360["text"] = "fermer"
        GButton_360.place(x=830,y=640,width=295,height=49)
        GButton_360["command"] = self.GButton_360_command

        GButton_688=tk.Button(root)
        GButton_688["bg"] = "#1e9fff"
        ft = tkFont.Font(family='Times',size=18)
        GButton_688["font"] = ft
        GButton_688["fg"] = "#f7f1f1"
        GButton_688["justify"] = "center"
        GButton_688["text"] = "Face Recognition (Notre deep Model)"
        GButton_688.place(x=830,y=430,width=293,height=50)
        GButton_688["command"] = self.GButton_688_command

        GButton_708 = tk.Button(root)
        GButton_708["bg"] = "#1e9fff"
        ft = tkFont.Font(family='Times', size=18)
        GButton_708["font"] = ft
        GButton_708["fg"] = "#f7f1f1"
        GButton_708["justify"] = "center"
        GButton_708["text"] = "MultiDetection-(Haar Cascade)"
        GButton_708.place(x=830, y=510, width=294, height=51)
        GButton_708["command"] = self.GButton_708_command

        """
        GLabel_464=tk.Label(root,image=imglogo)
        #GLabel_464["bg"] = "#d3b3b3"
        #ft = tkFont.Font(family='Times',size=10)
        #GLabel_464["font"] = ft
        #GLabel_464["fg"] = "#333333"
        GLabel_464["image"]=imglogo
        GLabel_464["justify"] = "center"
        GLabel_464["text"] = "label"

        GLabel_464.place(x=100,y=280,width=653,height=417)
        """
        label = Label(root, image=imglogo)
        label.image = imglogo

        label.place(x=100,y=280,width=653,height=417)
        #label.pack()

    def GButton_125_command(self):
        print("command")


    def GButton_360_command(self):
        print("command")


    def GButton_688_command(self):
        print("command")

    def GButton_708_command(self):
        print("command")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
