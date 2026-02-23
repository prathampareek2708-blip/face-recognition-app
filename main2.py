# ============================ IMPORTS ============================

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

# ============================ FUNCTIONS ============================

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)


def contact():
    mess.showinfo("Contact us", "Please contact us on : shubhamkumar8180323@gmail.com")


def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror("Missing File", "haarcascade_frontalface_default.xml not found.")
        window.destroy()


# ============================ PASSWORD SYSTEM ============================

def save_pass():
    assure_path_exists("TrainingImageLabel")
    path = "TrainingImageLabel/psd.txt"

    if os.path.isfile(path):
        with open(path, "r") as tf:
            key = tf.read()
    else:
        mess.showinfo("Info", "No password found. Set new password.")
        new_pass = tsd.askstring("New Password", "Enter new password", show='*')
        if new_pass:
            with open(path, "w") as tf:
                tf.write(new_pass)
        return

    if old.get() == key:
        if new.get() == nnew.get():
            with open(path, "w") as tf:
                tf.write(new.get())
            mess.showinfo("Success", "Password changed successfully")
            master.destroy()
        else:
            mess.showerror("Error", "New passwords do not match")
    else:
        mess.showerror("Error", "Wrong old password")


def change_pass():
    global master, old, new, nnew
    master = tk.Toplevel(window)
    master.geometry("400x160")
    master.title("Change Password")

    tk.Label(master, text="Old Password").pack()
    old = tk.Entry(master, show='*')
    old.pack()

    tk.Label(master, text="New Password").pack()
    new = tk.Entry(master, show='*')
    new.pack()

    tk.Label(master, text="Confirm Password").pack()
    nnew = tk.Entry(master, show='*')
    nnew.pack()

    tk.Button(master, text="Save", command=save_pass).pack()


# ============================ IMAGE CAPTURE ============================

def TakeImages():
    check_haarcascadefile()
    assure_path_exists("StudentDetails")
    assure_path_exists("TrainingImage")

    Id = txt.get()
    name = txt2.get()

    if not name.replace(" ", "").isalpha():
        message.configure(text="Enter Correct Name")
        return

    serial = 0
    csv_path = "StudentDetails/StudentDetails.csv"

    if os.path.isfile(csv_path):
        with open(csv_path, 'r') as f:
            serial = len(list(csv.reader(f))) // 2
    else:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
            serial = 1

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    sampleNum = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(
                f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg",
                gray[y:y + h, x:x + w]
            )
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Taking Images", img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if sampleNum > 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([serial, '', Id, '', name])

    message.configure(text=f"Images Taken for ID : {Id}")


# ============================ TRAIN MODEL ============================

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = getImagesAndLabels("TrainingImage")

    if not faces:
        mess.showerror("Error", "No images found")
        return

    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")

    message.configure(text="Profile Saved Successfully")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)

    return faces, Ids


# ============================ TRACK ATTENDANCE ============================

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
        mess.showerror("Error", "Please train model first")
        return

    recognizer.read("TrainingImageLabel/Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 50:
                name = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                name = str(name)[2:-2]
            else:
                name = "Unknown"

            cv2.putText(im, name, (x, y+h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Taking Attendance", im)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# ============================ GUI ============================

window = tk.Tk()
window.geometry("1280x720")
window.title("Attendance System")
window.configure(background='#2d420a')

clock = tk.Label(window, font=('comic', 22, ' bold '))
clock.pack()
tick()

txt = tk.Entry(window)
txt.pack()

txt2 = tk.Entry(window)
txt2.pack()

message = tk.Label(window, text="")
message.pack()

tk.Button(window, text="Take Images", command=TakeImages).pack()
tk.Button(window, text="Save Profile", command=TrainImages).pack()
tk.Button(window, text="Take Attendance", command=TrackImages).pack()
tk.Button(window, text="Change Password", command=change_pass).pack()
tk.Button(window, text="Exit", command=window.destroy).pack()

window.mainloop()