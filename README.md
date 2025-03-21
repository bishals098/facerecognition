# Face Recognition

This project demonstrates a simple face recognition system using OpenCV. The system captures video from the webcam, detects faces in real-time, and stores the detected faces as images.

## Prerequisites

- Python 3.x
- OpenCV
- Haarcascade Data File

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/bishals098/facerecognition.git
    cd facerecognition
    ```

2. Install the required packages:

    ```bash
    pip install opencv-python numpy
    ```

## Usage

1. Run the face recognition script:

    ```bash
    python facerecog.py
    ```

2. The script will start capturing video from your webcam. It will detect faces and display the number of faces captured in the top-left corner of the video frame.

3. To stop the script, press the `q` key.

## Code Explanation

The script performs the following steps:

1. Import the necessary libraries:

    ```python
    import cv2
    import numpy as np
    ```

2. Initialize the video capture and face detection:

    ```python
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ```

3. Check if the Haar Cascade XML file is loaded successfully:

    ```python
    if facedetect.empty():
        raise IOError('Failed to load Haar Cascade XML file.')
    ```

4. Initialize variables to store face data and frame index:

    ```python
    faces_data = []
    i = 0
    ```

5. Start capturing video frames and detecting faces:

    ```python
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            if x+w > frame.shape[1] or y+h > frame.shape[0]:
                continue  # Skip if the face rectangle is out of bounds

            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i = i + 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    ```

## Acknowledgments

- This project uses the [OpenCV](https://opencv.org/) library for computer vision tasks.
