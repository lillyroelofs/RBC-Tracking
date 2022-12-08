RBC-Tracking - Lilly Roelofs - 12/5/2020

Description of files/folders in RBC-Tracking Repo:

SuperProjectMain.m - This is the main script used to run the model and all associated functions. 

DevelopTrainingImages.m - This script produces the ground truth images using the hand-drawn annotations from the Image Labeler app. The images are created and placed into the correct directory (folder name = class label). Also, the images are all resized to match the same dimensions. 

PrimaryTrainingImages - This is the input data, split between 2 subfolders: Not_RBC, RedBloodCell. These contain images from each class (absence of RBC/presence of RBC), which is used to train the machine learning classifier. This is the product of DevelopTrainingImages.m. 

IsSameObjectFinal.m - Because the SVM classifier often detects the same cell several times (slight variations in location which cover the same object), I developed this function to combine windows which are in close proximity. The goal is to average the x,y coordinates to produce the most accurate location of the cell. The output is a matrix containing the final location coordinates and the final number of positions found (after combining). 

CorrectPosition.m - This function is used to determine which cell positions refer to the same cell between frames. This is important as we want to track each individual cell's position throughout the video, so we need to determine which cell is which. The input of this function is the positions detected in the previous frame, the positions detected in the following frame, and an arbitrary value for the amount of distance we anticipate the cells to move between frames. The output is a matrix containing the final cell positions (in order according to the cell numbering) as well as the number of labels found. 

AnnotatedImages - Sample of the output images from the model. These images show tracking annotations of red blood cell's (RBC's) from a single well. A screenshot per 100 frames is included.

