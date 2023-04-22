import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from tensorflow.keras.models import Sequential
import cv2
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import threading
from tkinter import filedialog, messagebox
import imghdr


emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(
    3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "angry", 1: "disgust", 2: "fear",
                3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}


cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dist = {0: cur_path+"/data/emojis/angry.png", 1: cur_path+"/data/emojis/disgusted.png",
              2: cur_path+"/data/emojis/fearful.png", 3: cur_path+"/data/emojis/happy.png",
              4: cur_path+"/data/emojis/neutral.png", 5: cur_path+"/data/emojis/sad.png",
              6: cur_path+"/data/emojis/surprised.png"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [4]
global frame_number

selected_file_path = ""


def choose_file():
    global selected_file_path
    selected_file_path = filedialog.askopenfilename(
        title="Choose file",
        filetypes=[("Video/Image files",
                    "*.mp4 *.avi *.mov *.mkv *.jpg * *.PNG *.jpeg *.png *.gif *.PNG *.webp")]
    )


def show_subject():
    cap1 = cv2.VideoCapture(
        r'D:\GUI-python\src\data\about2.jpg')
    global selected_file_path
    if not cap1.isOpened():
        print("Can't open the camera")
        messagebox.showinfo("Error", "Please choose a file")
        return

    # Determine the file type (image or video)
    _, file_extension = os.path.splitext(selected_file_path)
    if file_extension.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.PNG', '.webp'):
       
        frame = cv2.imread(selected_file_path)

        # Detect faces and emotions in the current frame
        bounding_box = cv2.CascadeClassifier(
            r'D:\GUI-python\src\data\Newfolder\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
            show_text[0] = maxindex

        
        cv2.imshow('Image', frame)
        cv2.waitKey(0)
    elif file_extension.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
        # Read the selected video
        cap1 = cv2.VideoCapture(selected_file_path)
        if not cap1.isOpened():
            print("Can't open the camera")
    global frame_number
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1
    if frame_number >= length:
        frame_number = 0 
    cap1.set(1, frame_number)
    flag1 = cap1.grab()
    flag1, frame1 = cap1.retrieve()
    frame1 = cv2.resize(frame1, (300, 400))
    bounding_box = cv2.CascadeClassifier(
        r'D:\GUI-python\src\data\New folder\haarcascade_frontalface_default.xml')
    
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-20), (x+w, y+h+10), (300, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
        # Check if the selected expression matches the detected expression
        if selected_expression == emotion_dict[maxindex]:
            cap1.release()  
            return
    if flag1 is None:
        print('Major Error!')
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
       
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
        cv2.destroyAllWindows()


def show_avatar():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    img2 = Image.fromarray(frame2)
    # Resize the image to 50% of its original size
    img2 = img2.resize((int(img2.width * 0.5), int(img2.height * 0.5)))
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk = imgtk2
    lmain3.configure(
        text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    lmain2.configure(image=imgtk2)
    root.update()
    # Call the show_avatar function with a unique name
    lmain2.after(10, show_avatar)


global selected_expression
selected_expression = ''


def drop_down_menu():
    options = [" ", "angry", "disgust", "fear",
               "happy", "neutral", "sad", "surprise"]

    # Create the label for the dropdown menu
    label = tk.Label(root, text="Select expression to match:",
                     font=("arial", 15), fg="white", bg="black")
    label.pack(side=TOP, padx=30, pady=10)

    var = tk.StringVar(root)
    var.set(options[0])
    dropdown = tk.OptionMenu(root, var, *options, command=on_select)
    dropdown.pack()


def on_select(option):
    global selected_expression
    selected_expression = option


def update_dropdown(value):
    print(value)


threshold = 50


def update_threshold(value):
    global threshold
    threshold = int(value)


if __name__ == '__main__':
    frame_number = 0

    # Initialize the window
    root = tk.Tk()
    root.title("Image Expression Recognizer, GUI ")
    root.geometry("1400x720+100+10")
    root['bg'] = 'black'

    # Create the dropdown menu
    options = ["angry", "disgust", "fear",
               "happy", "neutral", "sad", "surprise"]
    expression_label = tk.Label(
        root, text="Select expression to recognize:", fg="white", bg="black", font=("arial", 15))
    expression_label.pack(side=TOP, padx=30, pady=10)
    dropdown = ttk.Combobox(root, values=options, font=('arial', 15))
    dropdown.current(0)
    dropdown.bind("<<ComboboxSelected>>",
                  lambda event: on_select(dropdown.get()))
    dropdown.pack(side=TOP, padx=30, pady=10)

    # Create the slider
    slider_label = tk.Label(
        root, text="Threshold Confidence:", fg="white", bg="black", font=("arial", 15))
    slider_label.pack(side=TOP, padx=30, pady=10)
    slider_value = tk.DoubleVar()
    slider = tk.Scale(root, variable=slider_value, from_=0,
                      to=100, orient=HORIZONTAL, length=200, font=("arial", 15))
    slider.set(50)
    slider.pack(side=TOP, padx=30, pady=10)

    # Create the image labels
    lmain = tk.Label(master=root, padx=30, bd=5)
    lmain1 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=200)
    lmain3.pack()
    lmain3.place(x=900, y=150)
    lmain2 = tk.Label(master=root, bd=10)
    lmain2.pack(side=BOTTOM)
    lmain2.place(x=900, y=300)

    is_detecting = False

    def start_detection():
        global is_detecting
    while is_detecting:
        # Perform facial expression recognition here
        # If a facial expression is recognized, set is_detecting to False to stop the loop

        # Create the start/stop detection button
        is_detecting = False

    def stop_detection():
        # Add implementation to stop detection process
        pass

    def start_stop_detection():
        global is_detecting
        if not is_detecting:
            is_detecting = True
            detection_label.config(text="Detection in progress...")
            loading_label.config(text="...")
            threading.Thread(target=show_subject).start()
            threading.Thread(target=show_avatar).start()
            threading.Thread(target=start_detection).start()
        else:
            is_detecting = False
            detection_label.config(text="")
            loading_label.config(text="")
            stop_detection()

    detection_button = tk.Button(
        root, text="Start/Stop Detection", font=('arial', 20), command=start_stop_detection)
    detection_button.pack(side=TOP, padx=30, pady=10)

    # Create the detection label and loading label
    detection_label = tk.Label(
        root, text="", fg="white", bg="black", font=("arial", 15))
    detection_label.pack(side=TOP, padx=30, pady=10)

    loading_label = tk.Label(root, text="", fg="white",
                             bg="black", font=("arial", 30))
    loading_label.pack(side=TOP, padx=30, pady=10)

    # Create the exit button
    exitButton = Button(root, text='Quit', fg='red',
                        command=root.destroy, font=('arial', 25, 'bold'))
    exitButton.pack(side=BOTTOM)

    
    choose_file_button = tk.Button(
        root, text="Choose File for facial recognition", command=choose_file)
    choose_file_button.place(x=70, y=70)

    # Start the threads
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()

    # Start the main loop
    root.mainloop()
    
#hello thank you for trusting me and working with me, you are always welcome to work with me anytime
#make the reamining $50 via my paypal, wambuacharles33@gmail.com
#
#to run this project run:
                   # 1.open terminal on your vs code or whatever editor you are using
                   #2.navigate to src folder by, cd src
                   #3 run python emoji.py, to run
# i have trainned the model for 7 epochs, for better performance ou can change the number to a higher number, like 50, on train.py, once you make the change run python train.py                   
