# Academic Projects

This repository contains all projects and assignments that are done from Binghamton University and Georgia Technology of Institution. 

### Binghamton University
**2015 Undergraduate summer research internship: Real-time Robust Person Detection and Tracking**
- This project was focusing on sensor fusion with the depth sensor to improve facial detection efficiency.
- folder: "2015 summer intern"

### Georgia Technology of Institution

Class details: https://www.omscs.gatech.edu/cs-6476-computer-vision

This course provides an introduction to computer vision including fundamentals of image formation, camera imaging geometry, feature detection and matching, stereo, motion estimation and tracking, image classification and scene understanding. We’ll develop basic methods for applications that include finding known models in images, depth recovery from stereo, camera calibration, image stabilization, automated alignment, tracking, and recognition.

The focus of the course is to develop the intuitions and mathematics of the methods in lecture, and then to learn about the difference between theory and practice in the problem sets.

### Assignments ###
#### 1. Image as function ####

**Description**
This problem set is really just to make sure you can load an image, manipulate the values, produce some output, and submit the code along with the report.  Note that autograded problems will be marked with a (*).
It is expected that you have set up your environment properly. All problem sets will require the following libraries: NumPy, SciPy and OpenCV 2.4.13. 

**Learning Objectives**
- Learn to load, display, and save images.
- Study how images can be represented as functions.
- Identify the difference between an RGB and Monochrome / Grayscale images.
- Apply linear algebra concepts to manipulate image pixel values.
- Perform basic statistical operations in arrays.
- Introduce the concept of noise in an image.

#### 2. Detecting Traffic Signs and Lights ####

**Description**
Problem Set 2 is aimed at introducing basic building blocks of image processing.  Key areas that we wish to see you implement are: loading and manipulating images, producing some valued output of images, and comprehension of the structural and semantic aspects of what makes an image.  Relevant Modules are 1-2.

For this and future assignments, we will give you a general description of the problem.  It is up to the student to think about and implement a solution to the problem using what you have learned from the lectures and readings.  You will also be expected to write a report on your approach and lessons learned.

**Learning Objectives**
- Identify how images are represented using 2D and 3D arrays.
- Learn the representation of color channels in 3D arrays and the predominance of a certain color in an image.
- Use Hough tools to search and find lines and circles in an image.
- Use the results from the Hough algorithms to identify basic shapes.
- Understand how objects can be selected based on their pixel locations and properties.
- Address the presence of distortion / noise in an image.
- Identify what challenges real-world images present over simulated scenes.

#### 3. Introduction to AR ####
**Description**
Problem Set 3 introduces basic concepts behind Augmented Reality, using the contents that you will learn in
modules 3A-3D and 4A-4C: Projective geometry, Corner detection, Perspective imaging, and Homographies,
respectively.
Additionally, you will also learn how to read from a video, process each video frame by identifying important
features, insert images within images, and assemble a video from a sequence of frames.

**Learning Objectives**
- Find markers using circle and corner detection, convolution, and / or pattern recognition.
- Learn how projective geometry can be used to transform a sample image from one plane to another.
- Address the marker recognition problem when there is noise in the scene.
- Implement backwards (reverse) warping.
- Understand how video can be extracted in sequences of images, and replace specific areas of each
image with different content.
- Assemble a video from a sequence of images.

#### 4. Motion Detection ####
**Description**
Problem Set 4 introduces optic flow as the problem of computing a dense flow field where a flow field is a vector field <u(x,y), v(x,y)>. We discussed a standard method — Hierarchical Lucas and Kanade — for computing these vectors. This assignment will have you implement methods from simpler operations in order to understand more about array manipulation and the math behind them. We would like you to focus on movement in images, and frame interpolation, using concepts that you will learn from modules 6A-6B: Optic Flow.

**Learning Objectives**
- Implement the Lucas-Kanade algorithm based on the concepts learned from the lectures.
- Learn how pixel movement can be seen as flow vectors.
- Create image resizing functions with interpolation.
- Implement the Hierarchical Lucas-Kanade algorithm.
- Understand the benefits of using a Pyramidal approach.
- Understand the theory of action recognition.

#### 5. Object Tracking and Pedestrian Detection ####
**Description**
In this problem set you are going to implement tracking methods for image sequences and videos. The main algorithms you will be using are the Kalman and Particle Filters.

**Learning Objectives**
- Identify which image processing methods work best in order to locate an object in a scene.
- Learn to how object tracking in images works.
- Explore different methods to build a tracking algorithm that relies on measurements and a prior
state.
- Create methods that can track an object when occlusions are present in the scene.
