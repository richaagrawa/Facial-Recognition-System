Facial_Recognition_System

**Wondering how machine can recognise a Face, So you are at the right place. :)**

Before doing that let me tell you how this thing works. The idea of the task is to create a facial recognition system which will consist of broadly:

*Image/Video Input -> Face Detector -> Required Preprocessing -> Face Recognizer -> Results (Embedding) -> Comparison with faces in the database and publish the results*

There are two specific repositories that are implemented.
1) FaceBoxes  (https://arxiv.org/abs/1708.05234)(https://github.com/TropComplique/FaceBoxestensorflow)
2) Facenet (https://arxiv.org/abs/1503.03832) (https://github.com/davidsandberg/facenet)

But the different thing that is done here is : detector of Facenet is unpluged and replaced with the output from faceboxes.

