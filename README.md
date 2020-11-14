# FaceRecognizer
Application features:
- Creating a training dataset for face recognition systems
- Training a face recognition model using machine learning
- 4 face recognition algorithms: LBPH, Eigenfaces, Fisherfaces, Deep Learning
- Testing created face recognition system manually or using k-fold cross-validation

## Requirements: 
```sh
Python 3.7
```
\
Python libraries:
```sh
PyYaml
opencv-python
opencv-contrib-python (required for dlib face alignment)
numpy
kivy (kivy-sdl2, kivy-glew)
tensorflow (required by keras)
keras
scikit-learn
shutil
os
sklearn
pickle
dlib
```
\
Dlib requires:
```sh
Cmake
Visual Studio Build Tools with C++ libraries
```
\
for GPU support: 
```sh
CUDA 10.1
cuDNN
```
\
cuDNN installation: 
```sh
Copy [installpath]\cuda\bin\cudnn64_[version].dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v[version]\bin.\
Copy [installpath]\cuda\include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v[version]\include.\
Copy [installpath]\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v[version]\lib\x64.
```

<br />

#### Face recognition algorithms based on:
1. OpenCV FaceRecognizer class (<https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html>)
2. Scikit-learn SVM classifier (<https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>)
3. Keras FaceNet Pre-Trained Model (<https://github.com/nyoki-mtl/keras-facenet>)

<br />

#### Face detection and alignment based on:
1. Dlib face detector (<http://dlib.net/face_detector.py.html>)
2. Dlib face alignment (<http://dlib.net/face_alignment.py.html>)
3. Dlib 68 point facial landmark predictor (<http://dlib.net/face_landmark_detection.py.html>)
