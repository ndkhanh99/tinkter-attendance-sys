import pickle
import facenet.src.facenet as facenet
import argparse
from Detector import main_app
from create_classifier import train_classifer, regFaces
from create_dataset import start_capture
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage
from PIL import Image, ImageTk
import cv2
import os
import imutils
from Helper.align import detect_face
import datetime
import pytz
from datetime import timedelta

import tensorflow as tf
from imutils.video import VideoStream
import numpy as np
import random
import time
import serial
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter


cred = credentials.Certificate(
    "key/attendace-sys-firebase-adminsdk-e2nde-ea40d5feeb.json")
firebase_admin.initialize_app(cred)

global arduino
arduino = serial.Serial("/dev/cu.usbmodem14201", 9600, timeout=1)
time.sleep(0.2)  # wait for serial to open

names = set()
db = firestore.client()
utc = pytz.UTC


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        global open_webcam
        open_webcam = 'false'
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(
            family='Helvetica', size=16, weight="bold")
        self.title("Face Recognizer")
        self.resizable(False, False)
        app_width = 1200
        app_height = 600
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width / 2) - (app_width / 2)
        y = (screen_height / 2) - (app_height / 2)

        self.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor='center')
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageDetectFingert, PageFour, PageTakeFace, PageDetectFace):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            f = open("nameslist.txt", "a+")
            for i in names:
                f.write(i+" ")
            self.destroy()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        load1 = Image.open("Touch_ID.png")
        load1 = load1.resize((250, 250), Image.ANTIALIAS)
        render1 = PhotoImage(file='Touch_ID.png')

        render1 = ImageTk.PhotoImage(Image.open(
            "Touch_ID.png").resize((250, 250), Image.ANTIALIAS))

        button4 = tk.Button(
            self, image=render1, command=lambda: self.controller.show_frame("PageTwo"))
        button4.image = render1
        button4.grid(row=0, column=0, rowspan=4,
                     padx=10, pady=12, sticky="nsew")

        load2 = Image.open("face-id-id.png")
        load2 = load2.resize((50, 50), Image.ANTIALIAS)
        render2 = PhotoImage(file='face-id-id.png')

        render2 = ImageTk.PhotoImage(Image.open(
            "face-id-id.png").resize((250, 250), Image.ANTIALIAS))

        button5 = tk.Button(
            self, image=render2, command=lambda: self.controller.show_frame("PageOne"))
        button5.image = render2
        button5.grid(row=1, column=1, rowspan=4,
                     padx=10, pady=12, sticky="nsew")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            with open("nameslist.txt", "w") as f:
                for i in names:
                    f.write(i + " ")
            self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        color1 = '#020f12'
        color2 = '#05d7ff'
        color3 = '#65e7ff'
        color4 = 'BLACK'
        color5 = 'YELLOW'
        self.buttoncanc = tk.Button(self,
                                    background=color2,
                                    foreground=color4,
                                    activebackground=color3,
                                    activeforeground=color4,
                                    highlightthickness=2,
                                    highlightbackground=color2,
                                    width=15,
                                    height=2,
                                    border=0,
                                    cursor='hand2',
                                    text="Cancel",
                                    font=('Arial', 16, 'bold'),
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttoncanc.place(relx=0.5, rely=0.5, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="Add new",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageTakeFace"))
        self.buttonTakeFace.place(relx=0.5, rely=0.3, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="Detect",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageDetectFace"))
        self.buttonTakeFace.place(relx=0.5, rely=0.1, anchor='center')


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        color1 = '#020f12'
        color2 = '#05d7ff'
        color3 = '#65e7ff'
        color4 = 'BLACK'
        color5 = 'YELLOW'

        self.buttoncanc = tk.Button(self,
                                    background=color2,
                                    foreground=color4,
                                    activebackground=color3,
                                    activeforeground=color4,
                                    highlightthickness=2,
                                    highlightbackground=color2,
                                    width=15,
                                    height=2,
                                    border=0,
                                    cursor='hand2',
                                    text="Cancel",
                                    font=('Arial', 16, 'bold'),
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttoncanc.place(relx=0.5, rely=0.5, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="New Fingerprint",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageThree"))
        self.buttonTakeFace.place(relx=0.5, rely=0.3, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="Detect Finger",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageDetectFingert"))
        self.buttonTakeFace.place(relx=0.5, rely=0.1, anchor='center')


class PageThree(tk.Frame):

    global message
    message = False

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global names
        self.controller = controller

        def arduino_detect(msg):
            print('Running. Press CTRL-C to exit.')
            if message == True:
                arduino = serial.Serial(
                    "/dev/cu.usbmodem14201", 9600, timeout=1)
                time.sleep(0.1)  # wait for serial to open
                if arduino.isOpen():
                    print("{} connected!".format(arduino.port))
                    try:
                        # cmd=str(input("Enter command : "))
                        arduino.write(str(msg).encode())
                        while True:
                            time.sleep(0.5)  # wait for arduino to answer
                            answer = arduino.readline()
                            print(answer)
                            if answer == b'ok':
                                # print('break')
                                # cmd=str(input("Enter command : "))
                                return answer
                            # arduino.flushInput()
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt has been caught.")

        def arduino_enroll(msg, new_finger_id):
            print(msg)
            print(new_finger_id)
            print('Running. Press CTRL-C to exit.')
            with serial.Serial("/dev/cu.usbmodem14201", 9600, timeout=1) as enroll_finger:
                time.sleep(0.1)  # wait for serial to open
                # global arduino
                if enroll_finger.isOpen():
                    print("{} connected!".format(enroll_finger.port))
                    try:
                        while True:
                            time.sleep(0.5)  # wait for arduino to answer
                            answer = enroll_finger.readline()
                            print(answer)
                            # if b"Sensor contains" in answer or b"doesn't" in answer:
                            enroll_finger.write(str(msg).encode())
                            if answer == b'#id\r\n':
                                # print('break')
                                arduino.flushInput()
                                arduino.flushOutput()
                                arduino.reset_input_buffer()
                                arduino.reset_output_buffer()
                                # cmd = str(input("Enter command : "))
                                cmd = str(new_finger_id)
                                enroll_finger.write(str(cmd).encode())
                                while True:
                                    answer = enroll_finger.readline()
                                    print(answer)
                                    if answer == b'Place same finger again\r\n':
                                        messagebox.showinfo(
                                            'NOTIFY', 'Take out and Place same finger again!')
                                    if b'did not match' in answer:
                                        print('did not match')
                                        return
                                    if b'ID#' in answer:
                                        id_ = answer
                                        id_ = id_.decode(
                                            'utf-8').split("#")[1]
                                        print(id_)
                                    if answer == b'Stored!\r\n':
                                        db.collection('users').document(
                                            student_code_entry.get()).set({'finger_id': id_}, merge=True)
                                        db.collection(
                                            'fingers').document('id').update({'id': new_finger_id + 1})
                                        self.notify.config(
                                            text='Successfully encode fingerprint')
                                        break
                                    # return answer
                                break
                            # arduino.flushInput()
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt has been caught.")

        def start_enroll():
            stop_enroll()
            global message
            message == True
            result = db.collection('users').document(
                student_code_entry.get()).get()
            if result.exists:
                print(result.to_dict())
                if "finger_id" in result.to_dict():
                    messagebox.showinfo(
                        'ALERT', 'You already registered your fingerprint!')
                    print("Exists")
                    return
                else:
                    print("new finger register")
                    new_finger_id = db.collection(
                        'fingers').document('id').get()
                    new_finger_id = new_finger_id.to_dict()
                    new_finger_id = new_finger_id['id']
                    arduino_enroll(2, new_finger_id)
            else:
                print('no match')
                messagebox.showwarning('ALERT', 'No Users Match')
                return

        def stop_enroll():
            global message
            message == False

        label1 = tk.Label(
            self, text="Enter your student code to register new fingerprint")
        label1.place(relx=0.5, rely=0.1, anchor='center')

        student_code_entry = tk.Entry(self)
        student_code_entry.place(relx=0.5, rely=0.2, anchor='center')

        buttoncanc2 = tk.Button(self, text="Enroll Fingerprint", bg="#ffffff",
                                fg="#263942", command=start_enroll)
        buttoncanc2.place(relx=0.5, rely=0.3, anchor='center')

        self.notify = tk.Label(
            self, text="", foreground='green',)
        self.notify.place(relx=0.5, rely=0.4, anchor='center')

        buttoncanc1 = tk.Button(self, text="Cancel", bg="#ffffff",
                                fg="#263942", command=lambda: controller.show_frame("StartPage"))
        buttoncanc1.place(relx=0.5, rely=0.5, anchor='center')


class PageDetectFingert(tk.Frame):

    global message
    message = False

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global names
        global status
        status = 'loading'
        self.controller = controller

        def arduino_detect(msg, mode):
            print(msg)
            print('Running. Press CTRL-C to exit.')
            # with serial.Serial("/dev/cu.usbmodem14201", 9600, timeout=1) as arduino:
            #     time.sleep(0.1)  # wait for serial to open
            global arduino
            if arduino.isOpen():
                print("{} connected!".format(arduino.port))
                try:
                    while True:
                        time.sleep(0.5)  # wait for arduino to answer
                        answer = arduino.readline()
                        print(answer)
                        if b"doesn't" in answer:
                            messagebox.showinfo(
                                "ALERT", "No fingerprint to detect")
                            break
                        # if b'Sensor contains' in answer:
                        arduino.write(str(msg).encode())
                        arduino.flushInput()
                        while True:
                            answer = arduino.readline()
                            print(answer)
                            idText = b''
                            if b'Unknown' in answer or b'no match' in answer:
                                return
                            if b'ID' in answer:
                                idText = answer
                            if b'ok' in answer:
                                self.notify1.config(
                                    text='Successfully detect fingerprint')
                                finger_id = idText.decode(
                                    'utf-8').split("#")[1]
                                print(finger_id)
                                if finger_id != '':
                                    query = db.collection('users').where(
                                        "finger_id", "==", finger_id).get()
                                    for doc in query:
                                        user = doc.to_dict()
                                        result = db.collection(
                                            'current_subject').document('current').get()
                                        result = result.to_dict()
                                        today = datetime.datetime.now()
                                        status = 'in_time'
                                        if mode == 1:
                                            checkInExist = db.collection(
                                                'check_in')
                                            checkInExist = checkInExist.where(
                                                filter=FieldFilter('subject', '==', result['name']))
                                            checkInExist = checkInExist.where(
                                                filter=FieldFilter('finger_id', '==', finger_id))
                                            checkInExist = checkInExist.get()
                                            if len(checkInExist) != 0:
                                                messagebox.showwarning(
                                                    'ALERT', "You had checked in this subject")
                                                return

                                            time_compare = result['time_in'] + \
                                                timedelta(hours=7)
                                            if utc.localize(today) > time_compare:
                                                status = 'vao_tre'
                                                print('vao_tre')
                                            else:
                                                status = 'vao_dung_gio'
                                                print('vao dung gio')
                                            db.collection('check_in').add(
                                                {'subject': result['name'], 'student_id': user['student_id'], 'student_name': user['name'], 'status': status, 'type': 'fingerprint_detect', 'finger_id': user['finger_id'], 'time_in': today - timedelta(hours=7)})
                                        else:
                                            checkExist = db.collection('check_in').where(
                                                "finger_id", "==", finger_id).get()
                                            if len(checkExist) == 0:
                                                messagebox.showwarning(
                                                    'ALERT', "You haven't check in")
                                                return
                                            time_compare = result['time_out'] + \
                                                timedelta(hours=7)
                                            if utc.localize(today) > time_compare:
                                                status = 'ra_dung_gio'
                                                print(
                                                    'ra dung gio')
                                            else:
                                                status = 'ra_som'
                                                print('ra som')

                                            db.collection('check_out').add(
                                                {'subject': result['name'], 'student_id': user['student_id'], 'student_name': user['name'], 'status': status, 'type': 'fingerprint_detect', 'finger_id': user['finger_id'], 'time_out': today - timedelta(hours=7)})
                                break
                            # return answer
                        break
                except KeyboardInterrupt:
                    print("KeyboardInterrupt has been caught.")

        # finger checkin
        def start_check_in():
            stop_detect()
            global status
            status = 'Init Parameters'
            print(status)
            global message
            message == True
            arduino_detect(1, 1)

        # finger checkout
        def start_check_out():
            stop_detect()
            arduino_detect(1, 2)

        def stop_detect():
            global message
            message == False

        buttoncanc2 = tk.Button(self, text="Finger Check In", bg="#ffffff",
                                fg="#263942", command=start_check_in)
        buttoncanc2.place(relx=0.5, rely=0.2, anchor='center')

        buttoncanc2 = tk.Button(self, text="Finger Check Out", bg="#ffffff",
                                fg="#263942", command=start_check_out)
        buttoncanc2.place(relx=0.5, rely=0.3, anchor='center')

        self.notify1 = tk.Label(
            self, text=status)
        self.notify1.place(relx=0.5, rely=0.4, anchor='center')

        buttoncanc1 = tk.Button(self, text="Cancel", bg="#ffffff",
                                fg="#263942", command=lambda: controller.show_frame("StartPage"))
        buttoncanc1.place(relx=0.5, rely=0.5, anchor='center')


class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller


### PAGE TAKE FACES ###
num_of_images = 0


class PageTakeFace(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        global cam_on
        cam_on = False
        global cap
        cap = None
        # student_id = '_'.join(['student', stundent_id_entry])

        def display_frame():
            global cam_on
            global num_of_images
            detector = cv2.CascadeClassifier(
                "./data/haarcascade_frontalface_default.xml")

            id = stundent_id_entry.get()

            filepath = './data/student/raw/' + id

            isExist = os.path.exists(filepath)

            if not isExist:
                print('The new directory is created!')
                print(filepath)
                os.makedirs(filepath)

            if cam_on:

                ret, frame = cap.read()

                if ret:
                    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    filename = '.'.join([str(num_of_images), 'jpg'])
                    path = os.path.join(filepath, filename)
                    cv2.imwrite(path, frame)
                    face = detector.detectMultiScale(
                        image=opencv_image, scaleFactor=1.1, minNeighbors=5)
                    for x, y, w, h in face:
                        cv2.rectangle(frame, (x, y),
                                      (x+w, y+h), (8, 238, 255), 2)
                        cv2.putText(frame, "Face Detected", (x, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (8, 238, 255))
                        cv2.putText(frame, str(str(num_of_images)+" images captured"),
                                    (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (8, 238, 255))
                        # new_img = frame[y:y+h, x:x+w]

                    # Capture the latest frame and transform to image
                    captured_image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))

                    # Convert captured image to photoimage
                    photo_image = ImageTk.PhotoImage(
                        captured_image.resize((500, 300), Image.ANTIALIAS))

                    # Displaying photoimage in the label
                    label_widget.photo_image = photo_image

                    # Configure image in the label
                    label_widget.configure(image=photo_image)

                    # Repeat the same process after every 10 seconds
                    num_of_images += 1

                    if num_of_images == 21:
                        stop_vid()
                        num_of_images = 0
                        db.collection('users').document(id).set(
                            {'name': first_name_entry.get(), 'student_id': id, 'image_path': filepath})
                        messagebox.showinfo(
                            "INSTRUCTIONS", "We captured 20 pic of your Face.")
                        return 'ok'

                label_widget.after(10, display_frame)

        def start_vid():
            global cam_on, cap
            stop_vid()
            cam_on = True
            cap = cv2.VideoCapture(1)
            display_frame()

        def stop_vid():
            label_widget.configure(image=None)
            label_widget.configure(image="")

            global cam_on
            cam_on = False

            if cap:
                cap.release()

        def trainmodel():
            # if self.controller.num_of_images < 300:
            #     messagebox.showerror(
            #         "ERROR", "No enough Data, Capture at least 300 images!")
            #     return
            regFaces()
            messagebox.showinfo(
                "SUCCESS", "You can now implement your detection")

        ####### take face screen #######

        user_info_frame = tk.LabelFrame(self, text="User Information")
        user_info_frame.grid(row=0, column=0, padx=20, pady=10)

        first_name_label = tk.Label(user_info_frame, text="Your Name")
        first_name_label.grid(row=0, column=0)
        student_id_label = tk.Label(user_info_frame, text="Your Student Id")
        student_id_label.grid(row=0, column=1)

        first_name_entry = tk.Entry(user_info_frame)
        stundent_id_entry = tk.Entry(user_info_frame)
        first_name_entry.grid(row=1, column=0)
        stundent_id_entry.grid(row=1, column=1)

        buttoncanc = tk.Button(user_info_frame, text="Cancel", bg="#ffffff",
                               fg="#263942", command=lambda: controller.show_frame("StartPage"))
        buttoncanc.grid(row=2, column=0, pady=10, ipadx=5, ipady=4)

        buttoncanc = tk.Button(user_info_frame, text="Confirm", bg="#ffffff",
                               fg="#263942", command=start_vid)
        buttoncanc.grid(row=2, column=1, pady=10, ipadx=5, ipady=4)

        label_widget = tk.Label(self)
        label_widget.grid(row=3, column=0)

        # age_label = tk.Label(user_info_frame, text="Age")
        # age_spinbox = tk.Spinbox(user_info_frame, from_=18, to=110)
        # age_label.grid(row=2, column=0)
        # age_spinbox.grid(row=3, column=0)

        for widget in user_info_frame.winfo_children():
            widget.grid_configure(padx=10, pady=5)

        button1 = tk.Button(self, text="Training Model",
                            command=trainmodel)
        button1.grid(row=8, column=0, padx=10,
                     pady=10, ipadx=5, ipady=4)


MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'model/facemodel.pkl'
FACENET_MODEL_PATH = 'model/20180402-114759.pb'
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")
with tf.Graph().as_default():

    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():

        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)

        # Get input and output tensors
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph(
        ).get_tensor_by_name("phase_train:0")

        pnet, rnet, onet = detect_face.create_mtcnn(
            sess, "Helper/align")


### PAGE DETECT FACES ###
class PageDetectFace(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        global cam_detect_on
        cam_detect_on = False
        global cap_detect
        cap_detect = None
        global count_unknown
        count_unknown = 0
        global detect_time
        detect_time = 0
        global mode
        mode = 1
        # student_id = '_'.join(['student', stundent_id_entry])

        def detect_frame():
            global cam_detect_on
            global count_unknown
            global detect_time
            if cam_detect_on:

                ret, frame = cap_detect.read()

                if ret:
                    # frame = cap.read()
                    frame = imutils.resize(frame, width=600)
                    frame = cv2.flip(frame, 1)
                    # rand_name = random.randint(0, 1000)
                    bounding_boxes, _ = detect_face.detect_face(
                        frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                    faces_found = bounding_boxes.shape[0]

                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=(np.int32))
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        print(bb[i][3] - bb[i][1])
                        print(frame.shape[0])
                        print((bb[i][3] - bb[i][1]) / frame.shape[0])
                        if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                            cropped = frame[bb[i][1]:bb[i]
                                            [3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(
                                cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=(cv2.INTER_CUBIC))
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(
                                -1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {
                                images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(
                                embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(
                                predictions, axis=1)
                            best_class_probabilities = predictions[(
                                np.arange(len(best_class_indices)), best_class_indices)]
                            best_name = class_names[best_class_indices[0]]
                            # print('Name: {}, Probability: {}'.format(
                            #     best_name, best_class_probabilities))

                            count_unknown += 1
                            if best_class_probabilities > 0.9:
                                image_path = 'images/attendance/'
                                print('Name: {}, Probability: {}'.format(
                                    best_name, best_class_probabilities))
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0,
                                                                                                  255,
                                                                                                  0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.putText(frame, best_name, (text_x, text_y), (cv2.FONT_HERSHEY_COMPLEX_SMALL), 1,
                                            (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, (str(round(best_class_probabilities[0], 3))), (text_x, text_y + 17), (cv2.FONT_HERSHEY_COMPLEX_SMALL),
                                            1, (255, 255, 255), thickness=1, lineType=2)

                                file_name = best_name + ".jpg"
                                if detect_time == 2:
                                    cv2.imwrite(os.path.join(
                                        image_path, file_name), frame)
                                    cv2.destroyAllWindows()
                                    result = db.collection(
                                        'current_subject').document('current').get()
                                    result = result.to_dict()
                                    user = db.collection(
                                        'current_subject').document('current').get()
                                    user.to_dict()
                                    today = datetime.datetime.now()
                                    print(today)
                                    status = 'in_time'
                                    global mode
                                    if mode == 1:
                                        time_compare = result['time_in'] + \
                                            timedelta(hours=7)
                                        print(time_compare)
                                        if utc.localize(today) > time_compare:
                                            status = 'vao_tre'
                                            print('vao tre')
                                        else:
                                            status = 'vao_dung_gio'
                                            print('vao dung gio')
                                        db.collection('check_in').add(
                                            {'subject': result['name'], 'student_id': best_name, 'student_name': 'name', 'status': status, 'type': 'face_detect', 'finger_id': user['finger_id'], 'time_in': today - timedelta(hours=7)})
                                        print('complete detect')
                                    else:
                                        time_compare = result['time_out'] + \
                                            timedelta(hours=7)
                                        if utc.localize(today) > time_compare:
                                            status = 'ra_dung_gio'
                                            print('ra dung gio')
                                        else:
                                            status = 'ra_som'
                                            print('ra som')
                                        db.collection('check_out').add(
                                            {'subject': result['name'], 'student_id': best_name, 'student_name': 'name', 'status': status, 'type': 'face_detect', 'time_out': today - timedelta(hours=7)})
                                        print('complete detect')
                                    time.sleep(1)
                                    stop_detect()
                                    detect_time = 0
                                    return best_name

                                detect_time += 1
                            else:
                                print('Unknown')
                                print(count_unknown)
                                if count_unknown == 20:
                                    print('break')
                                    best_name = 'unknown'
                                    # VideoStream(src=0).stop()
                                    # cap.stop()
                                    cv2.destroyAllWindows()
                                    stop_detect()
                                    return best_name

                    # Capture the latest frame and transform to image
                    captured_image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))

                    # Convert captured image to photoimage
                    photo_image = ImageTk.PhotoImage(
                        captured_image.resize((500, 300), Image.ANTIALIAS))

                    # Displaying photoimage in the label
                    detect_widget.photo_image = photo_image

                    # Configure image in the label
                    detect_widget.configure(image=photo_image)

                detect_widget.after(10, detect_frame)

        def start_check_in():
            global cam_detect_on, cap_detect, mode
            stop_detect()
            cam_detect_on = True
            cap_detect = cv2.VideoCapture(1)
            mode = 1
            detect_frame()

        def start_check_out():
            global cam_detect_on, cap_detect, mode
            stop_detect()
            cam_detect_on = True
            cap_detect = cv2.VideoCapture(1)
            mode = 2
            detect_frame()

        def stop_detect():
            detect_widget.configure(image=None)
            detect_widget.configure(image="")

            global cam_detect_on
            cam_detect_on = False

            if cap_detect:
                cap_detect.release()

        ####### detect face screen #######

        buttoncanc1 = tk.Button(self, text="Cancel", bg="#ffffff",
                                fg="#263942", command=lambda: controller.show_frame("StartPage"))
        buttoncanc1.place(relx=0.5, rely=0.5, anchor='center')

        buttoncanc3 = tk.Button(self, text="Stop Detect", bg="#ffffff",
                                fg="#263942", command=stop_detect)
        buttoncanc3.place(relx=0.5, rely=0.3, anchor='center')

        buttoncanc2 = tk.Button(self, text="Face Check In", bg="#ffffff",
                                fg="#263942", command=start_check_in)
        buttoncanc2.place(relx=0.5, rely=0.1, anchor='center')

        buttoncanc2 = tk.Button(self, text="Face Check Out", bg="#ffffff",
                                fg="#263942", command=start_check_out)
        buttoncanc2.place(relx=0.5, rely=0.2, anchor='center')

        detect_widget = tk.Label(self)
        detect_widget.place(relx=0.5, rely=0.7, anchor='center')


app = MainUI()
app.iconphoto(False, tk.PhotoImage(file='icon.ico'))
app.mainloop()
