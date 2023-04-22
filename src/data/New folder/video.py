def show_subject():
    cap1 = cv2.VideoCapture(
        r'D:\GUI-python\src\data\WhatsApp Video 2023-04-17 at 22.12.05.mp4')
    if not cap1.isOpened():
        print("Can't open the camera")
    global frame_number
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 1
    if frame_number >= length:
        frame_number = 0  # reset frame_number to zero
    cap1.set(1, frame_number)
    flag1 = cap1.grab()
    flag1, frame1 = cap1.retrieve()
    frame1 = cv2.resize(frame1, (300, 400))
    bounding_box = cv2.CascadeClassifier(
        r'D:\GUI-python\src\data\New folder\haarcascade_frontalface_default.xml')

    # Remove this line to keep the original color of the image
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
            cap1.release()  # Release the video capture and stop playing the video
            return

    if flag1 is None:
        print('Major Error!')
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        # Remove this line to keep the original color of the image
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

