# FaceRecognizer
Application features:
- creating a training dataset for face recognition systems
- train a face recognition model, using machine learning
- 4 face recognition algorithms: LBPH, Eigenfaces, Fisherfaces, Deep Learning
- testing created face recognition system

## Requirements: 
```sh
Python 3.7
```
\
Python libraries:
```sh
PyYaml
opencv-python
opencv-contrib-python
numpy
kivy (kivy-sdl2, kivy-glew)
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
MicrosoftVisualStudio with C++ libraries
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

###Face recognition algorithms based on:
1. OpenCV FaceRecognizer class (<https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html>)
2. Scikit-learn SVM classifier (<https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>)
3. Keras FaceNet Pre-Trained Model (<https://github.com/nyoki-mtl/keras-facenet>)
