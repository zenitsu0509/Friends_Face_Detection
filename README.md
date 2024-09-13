Real-Time Image Capture and Processing in Colab
===============================================

This project demonstrates how to capture real-time images using a webcam in Google Colab, convert them to grayscale, and save the processed image. The project leverages **Roboflow** for dataset management, **Ultralytics** for YOLO model integration, and **Google Colab** for an interactive development environment.

Features
--------

-   Capture real-time images using your webcam in a Colab notebook.
-   Convert the captured image to grayscale for further processing.
-   Integrate with YOLO (via Ultralytics) for real-time object detection (future extension).
-   Use Roboflow for dataset management and labeling.

Tech Stack
----------

-   **Roboflow**: Dataset collection and management.
-   **Ultralytics YOLO**: Object detection and real-time prediction (future extension).
-   **Colab**: Interactive environment for running Python scripts and capturing images.

Getting Started
---------------

1.  **Clone the Repository**\
    Clone this repository to your Google Colab environment or local machine.

    bash

    Copy code

    `git clone <repository_url>`

2.  **Install Dependencies** You need to install the following dependencies in your Colab notebook:

    bash

    Copy code

    `!pip install opencv-python-headless
    !pip install ultralytics`

3.  **Import Necessary Libraries** Import the required libraries for webcam capture and image processing:

    python

    Copy code

    `import cv2
    from google.colab.output import eval_js
    from IPython.display import Javascript, display
    from base64 import b64decode`

4.  **Run the Code** Use the `take_photo` function to capture images from the webcam, convert them to black-and-white, and save them:

    python

    Copy code

    `def take_photo(filename='photo.jpg', quality=0.8):
        # JavaScript code to access the webcam
        # ...

        # Capture the image
        # Convert to grayscale
        # Save as a .jpg file`

5.  **Integrate YOLO for Object Detection**\
    Extend this code by integrating the YOLO model from Ultralytics to add real-time object detection capabilities.

Example
-------

Here's a screenshot of the grayscale image captured and saved from the webcam:

Future Work
-----------

-   Add YOLOv9 for real-time object detection.
-   Integrate Roboflow for automatic labeling and dataset management.
-   Add functionality for real-time video stream processing.
