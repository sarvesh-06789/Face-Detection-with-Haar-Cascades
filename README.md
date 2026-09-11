# Face-Detection-with-Haar-Cascades
# Aim
To write a Python program using OpenCV to perform the following image processing and computer vision operations:

1. Extract a Region of Interest (ROI) from an image.

2. Perform face detection on static images (including single person, with glasses, and group photos) using Haar Cascades.

3. Perform eye detection within detected facial regions using Haar Cascades.

4. Perform real-time face detection with dynamic bounding boxes and labels on a live webcam feed using Matplotlib for display within a Jupyter Notebook.

# Algorithm
1. ROI Extraction
* Load the input image using cv2.imread().

* Convert the image color space from BGR to RGB using cv2.cvtColor() for correct color visualization in Matplotlib.

* Slice the image matrix using pixel coordinates img[y_start:y_end, x_start:x_end] to crop the designated Region of Interest.

* Plot the original image alongside the cropped ROI using Matplotlib subplots.

2. Static Face & Eye Detection
* Load the pre-trained Haar Cascade XML classifiers (haarcascade_frontalface_default.xml and haarcascade_eye.xml).

* Read the test image and convert it to grayscale (cv2.cvtColor() with COLOR_BGR2GRAY or cv2.imread() with flag 0).

* Pass the grayscale image to detectMultiScale() to retrieve bounding box coordinates (x, y, w, h) for each detected face.

* Draw bounding rectangles and text labels on the original image using cv2.rectangle() and cv2.putText().

* For eye detection, isolate the sub-region (ROI) corresponding to each face, apply the eye classifier detectMultiScale() within that sub-region, and draw bounding boxes around the eyes.

* Display the final annotated images using Matplotlib.

3. Real-Time Webcam Face Detection
* Initialize the video capture stream using cv2.VideoCapture(0).

* Continuously read frame-by-frame in a while loop.

* Convert each frame to grayscale and execute detectMultiScale() on the face classifier.

* Overlay bounding rectangles and label text ('Face Detected') on the RGB frame for each detected face.

* Render the updated frame inside the Jupyter Notebook using Matplotlib and call IPython.display.clear_output(wait=True) for smooth dynamic updates.

* Gracefully stop video capture upon user interruption (KeyboardInterrupt) and release the camera resource via cap.release().

# Program
```
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import urllib.request

# ==========================================
# 1. DOWNLOAD REQUIRED CASCAde & TEST FILES
# ==========================================
files_to_download = {
    'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/Bindhujaa02/Face-Detection-with-Haar-Cascades/main/haarcascade_frontalface_default.xml',
    'haarcascade_eye.xml': 'https://raw.githubusercontent.com/Bindhujaa02/Face-Detection-with-Haar-Cascades/main/haarcascade_eye.xml',
    'image_02.png': 'https://raw.githubusercontent.com/Bindhujaa02/Face-Detection-with-Haar-Cascades/main/image_02.png',
    'image_03.png': 'https://raw.githubusercontent.com/Bindhujaa02/Face-Detection-with-Haar-Cascades/main/image_03.png',
    'sample.jpg': 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&auto=format&fit=crop'
}

for filename, url in files_to_download.items():
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

# ==========================================
# 2. EXTRACT ROI FROM AN IMAGE
# ==========================================
img = cv2.imread('sample.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define ROI coordinates [y1:y2, x1:x2]
roi = img_rgb[100:400, 150:450]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(roi)
axes[1].set_title("Extracted ROI")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# ==========================================
# 3. FACE AND EYE DETECTION ON STATIC IMAGES
# ==========================================
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

withglass = cv2.imread('image_02.png', 0)
group = cv2.imread('image_03.png', 0)

def detect_face(img, scaleFactor=1.1, minNeighbors=5):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(detect_face(withglass), cmap='gray')
axes[0, 0].set_title("Faces in With Glasses Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(detect_face(group), cmap='gray')
axes[0, 1].set_title("Faces in Group Image")
axes[0, 1].axis("off")

axes[1, 0].imshow(detect_eyes(withglass), cmap='gray')
axes[1, 0].set_title("Eyes in With Glasses Image")
axes[1, 0].axis("off")

axes[1, 1].imshow(detect_eyes(group), cmap='gray')
axes[1, 1].set_title("Eyes in Group Image")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

# ==========================================
# 4. REAL-TIME FACE DETECTION VIA WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frame_rgb, 'Face Detected', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        plt.figure(figsize=(8, 6))
        plt.imshow(frame_rgb)
        plt.title(f"Real-Time Face Detection | Faces Count: {len(faces)}")
        plt.axis("off")
        plt.show()

        clear_output(wait=True)

except KeyboardInterrupt:
    print("Video stream stopped by user.")

finally:
    cap.release()
```
# Output
<img width="1192" height="940" alt="image" src="https://github.com/user-attachments/assets/d52a32cd-46a7-471d-b5c0-e4ce800daa4b" />

<img width="692" height="897" alt="Screenshot 2026-09-11 103053" src="https://github.com/user-attachments/assets/d59adf4a-0729-4333-a520-e388ed137f19" />

<img width="1122" height="905" alt="image" src="https://github.com/user-attachments/assets/040b1391-29ec-495f-9153-b3fba61b574d" />

# Result
1. ROI Extraction:

Specific regions of interest were successfully cropped by slicing array indices from the image matrix, verifying matrix manipulation in OpenCV.

2. Static Image Face & Eye Detection:

* Frontal Faces / Glasses: Frontal face detection accurately located faces across single-subject images.
* Eye detection worked well on clear eyes, though specular reflections or dark frames on eyeglasses occasionally caused false negatives or stray bounding boxes.
* Group Images: Faces angled directly toward the camera were detected. However, small, occluded, or side-profile faces required fine-tuning parameters such as scaleFactor and minNeighbors.

3. Real-Time Webcam Stream:

Real-time webcam frames were continuously retrieved, processed through the cascade model, annotated with bounding boxes and text labels, and smoothly displayed in Jupyter Notebook using matplotlib and clear_output.
