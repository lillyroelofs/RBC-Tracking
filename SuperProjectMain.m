%% Super Project - Tracking RBCs
% Lilly Roelofs 

clc 
clear 
close all

disp('Super Project - Lilly Roelofs')
disp(' ')

%% Problem 1, Step 1: Upload the video 

%Load the video: 
%video=VideoReader('Cells-in-wells.avi');
load('videoreader.mat')

%The "important" properties of the video are as follows:
%Duration: 24.4800
%NumFrames: 1224 - The video counts each half second up until 612 seconds, therefore each frame corresponds exactly to one half second
%Width: 1018
%Height: 769      

disp('Problem 1, Step 1: The videoreader has been created.')
disp(' ')

%% Problem 1, Step 2: Separate the video into each one of it's frames 

%First, determine how many frames there are:
num_frames=video.NumFrames; 

%Now, run through the each frame in a loop and save it as it's own image file, 
% and place all of them into a new folder, called 'RBC_image_collection'.

disp('Problem 1, Step 2: A folder entitled "RBC_image_collection" has been created which')
disp('contains an image of each individual frame in the video. It should be organized')
disp('sequentially and found in the current folder.')
disp(' ')

folder_name='RBC_image_collection';
if exist(folder_name,'dir')   %if the folder already exists, do not overwrite it 
    return
else
    mkdir(folder_name)   %if it doesn't exist, create a new folder directory  
    for indiv_frame=1:num_frames
        frames=read(video,indiv_frame);
        frame_fix=frames(:,:,1);   %only take one of the "layers" - do not want all 3                       
        destination=fullfile(folder_name,[(int2str(indiv_frame)),'.jpg']); %create destination with file name ("place holders")
        imwrite(frame_fix,destination);  %actually write and place the file into the destination
    end
end

%% Problem 1, Step 3: Develop Training Images for Classifier

% Because this section is very long and takes a few minutes to run, I took 
% it out of the main script and placed it into it's own .m file,
% "DevelopTrainingImages.m". This was feasible since the main output of
% this section is the PrimaryTrainingImages folder, which I can call on
% separately. There is a detailed explanation of how I 
% developed, extracted, and labeled images from the video in that script. 

% I still have included a few things that were written in this section as
% they are needed later in this script, which are presented below. 

% Need to define the image label names (label1 & label2)
image_label_data=load("gTruth1.mat"); 
gTruth=image_label_data.gTruth;
[im_datastore,boxlabel_datastore]=objectDetectorTrainingData(gTruth);
label1=gTruth.LabelDefinitions{1,1}{1,1};
label2=gTruth.LabelDefinitions{2,1}{1,1};

% Need to determine the size of the images 
resized_training_images=imageDatastore('PrimaryTrainingImages','IncludeSubfolders',1,'LabelSource','foldernames');
tot_num_images=numel(resized_training_images.Files);
% Determine the average size of all of the images
pixel_len=0;
pixel_height=0;
for image_num=1:tot_num_images
    temp_img=readimage(resized_training_images,image_num);
    temp_size=size(temp_img);
    pixel_len=pixel_len+temp_size(1);
    pixel_height=pixel_height+temp_size(2);
end
avg_len=round(pixel_len/tot_num_images);    
avg_height=round(pixel_height/tot_num_images);

% There is likely a easier way of doing this ^, but this was convenient for
% this section as I had already written the code. 

disp('Problem 1, Step 3: A set of all training images which have been resized to the same dimesions')
disp('can be found in the "PrimaryTrainingImages" folder. This folder has 2 subfolders which serve as the')
disp('labels for the images. In this case, the two subfolders are "RedBloodCell" and "Not_RBC".')
disp('These images and labels were created using the ImageLabeler App in MATLAB and were exported as')
disp('ground truth files into the folder for use.')
disp(' ')

%% Problem 1, Step 4: Extract Features from the Training Data Using HOG

% Quick Note: Generally, when people are using machine learning tools they
% create both training and testing sets before applying their classifier to
% the actual data in order to evaluate how well their model is working and 
% adjust from there. Because collecting the images was very time consuming 
% and difficult, I decided to informally evaluate my data instead by adjusting 
% the hyperparameters and methods based on how I visually saw the classifier
% working through the a few of the video frames. 

% First, I will show an example of how the HOG feature extractions works

% Open the new image set
resized_training_images=imageDatastore('PrimaryTrainingImages','IncludeSubfolders',1,'LabelSource','foldernames');

% Need to choose a few of HOG's hyperparameters, such as cell size, number 
% of bins, and block size. After various informal tests I found that the 
% following parameters produced the best feature extraction results:
cell_size=[4 4];   
num_bins=12;
block_size=[4 4];

% Test the feature extractor on a random image of an RBC 
test_image=readimage(resized_training_images,400);
[test_hog,test_vis]=extractHOGFeatures(test_image,'CellSize',cell_size,'NumBins',num_bins,'BlockSize',block_size);

%Plot the image to visualize how the feature extraction works
figure(1)
imshow(test_image)
hold on
plot(test_vis)

% Now we will extract the features of all of our training images

% Make a zero matrix of the correct size to store the features
hog_size=length(test_hog);
tot_trainingfeatures=zeros(tot_num_images,hog_size,'single');

% Create a for loop and run each image through the extractor to create a
% large training feature matrix (tot_trainingfeatures)
for image_num=1:tot_num_images
    temp_img=readimage(resized_training_images,image_num);
    temp_img=imbinarize(temp_img);
    tot_trainingfeatures(image_num,:)=extractHOGFeatures(temp_img,'CellSize',cell_size,'NumBins',num_bins,'BlockSize',block_size);
end

disp('Problem 1, Step 4: HOG (Histogram of Oriented Gradients) feature extraction has been performed on all')
disp('training images, and the results have been placed into one matrix called "tot_trainingfeatures".')
disp('Also, in a separate window a figure should open which displays a random test image with its HOG')
disp('features drawn over the image. This is a great representation of how HOG extraction works.') 
disp(' ')

%% Problem 1, Step 5: Train an SVM Classifier 

% Assign labels to the features 
training_labels=resized_training_images.Labels;

% Run this through an SVM classifier 
SVM_Classifier1=fitcecoc(tot_trainingfeatures,training_labels);

% I did attempt to improve the classifier by applying an ECOC to help
% eliminate error, but after multiple attempts it did not seem to affect
% the results much at all, so I stayed with a simple support vector machine. 

disp('Problem 1, Step 5: An SVM (Support Vector Machine) classifier has been applied to the')
disp('features extracted from the training images in the previous step. This classifier')
disp('determines whether or not an image contains a red blood cell.')
disp(' ')

%% Problem 1, Step 6: Run the Classifier on a Test Image

% First, I determined the dimensions for multiple wells I would like to observe.
% While working on this model, I tested it on several wells to determine how effective 
% and flexible the model is. Since I already had the coordinates written
% out, I kept them in the script. 
% Translating the coordinates to [xtopleft, ytopleft, x_width, y_height] using 
% some basic algebra, the following are the dimensions:
specific_frame1=[154,208,188,176];
specific_frame2=[154,408,188,176];
specific_frame3=[755,408,188,176];
specific_frame4=[357,208,188,176];
specific_frame5=[357,408,188,176];

% While my program works to some degree on all of these wells, it definitely performs 
% the most accurately  on the well found at specific_frame2, so I will be testing 
% the model on this well throughout the rest of this script.  
current_frame=specific_frame2;

% Test the classifier on the first frame of the video at the specified well
rbc_des=fullfile(folder_name,'1.jpg');
first_frame=imread(rbc_des);
new_fframe=imcrop(first_frame,current_frame);  %must crop the image to only include the well

% Must use a "sliding" method to run through the well at several points with the  
% same window size as the training images. This sliding method will run over 
% the entire image to ensure that all RBC's are detected. 
stepsize=5;    %this is the translation between windows
box_height=avg_len;    %window height
box_width=avg_height;    %window width

% Size of the image we are sliding over
[fframe_height,fframe_width]=size(new_fframe);

% Total number of steps
v_steps=floor(((fframe_height-box_height)/stepsize)); %vertical steps  
h_steps=floor(((fframe_width-box_width)/stepsize));  %horizontal steps 

% Now, we run through a loop to evaluate the SVM classifier at each of
% these image windows
col_temp=1;   %starting x position
row_temp=1;    %starting y position
RBC_position=[];   
for num_row_step=1:v_steps 
    for num_col_step=1:h_steps  
        temp_window=new_fframe(row_temp:row_temp+(box_width-1),col_temp:col_temp+(box_height-1));  %crop image at window
        temp_features=extractHOGFeatures(temp_window,'CellSize',cell_size,'NumBins',num_bins,'BlockSize',block_size);  %extract HOG features per window
        [prediction,score]=predict(SVM_Classifier1,temp_features);     %determine if window contains RBC    
        if prediction=='RedBloodCell'
            frame_position=[(col_temp+(current_frame(1))),(row_temp+(current_frame(2))),...
                box_width,box_height];  %this is the [x_leftcorner,y_leftcorner,width,height] value
            RBC_position=[RBC_position; frame_position];   %matrix containing all windows (before clustered together)
        end
        col_temp=col_temp+stepsize; %move a step to the right until it reaches the edge of the image
    end
    row_temp=row_temp+stepsize; % move a step down the image 
    col_temp=1;    %start back at the left side of the image
end

% Want to detect each object only once. Therefore, I created a user defined
% function to combine windows that are in close proximity (so multiple
% windows are not categorizing one cell)
[RBC_position_final,num_images_final]=IsSameObjectFinal(RBC_position,stepsize);

% Note: RBC_position_final contains a vector of the coordinates of the 
% clustered, final detector windows.

% Now we neeed to categorize the cells in the image by giving them a numbered label. Here, 
% we are going to assume that the classifier correctly found and labeled all 
% RBC's without adding additional, blank labels in the FIRST frame. This is
% very important and MUST be true for proper detection and tracking throughout
% the rest of the movie. The first frame provides the basis for the rest of
% the frames. 
firstframe_len=height(RBC_position_final); %correlates to the number of cells 
firstframe2=[]; %will hold the position vectors + the number of the cell label 
cell_names=[];  %will hold the name of each cell, in the same order as the firstframe2 variable
for num=1:firstframe_len   
    temp_pos=RBC_position_final(num,:);
    temp_name=['Cell',int2str(num)];
    cell_names=[cell_names;temp_name];
    temp_pos=[temp_pos,num];   %the last index of the vector (in position 5) correlates to the cell number label
    firstframe2=[firstframe2;temp_pos]; 
end

% Lastly, display the image with the windows which are numbered (labeling
% which cell is which)
annotated_frame=insertObjectAnnotation(first_frame,'Rectangle',firstframe2(:,1:4),firstframe2(:,5));
figure(2)
imshow(annotated_frame)

disp('Problem 1, Step 6: In a separate window, a new figure should open up which displays the first')
disp('frame of the video with the RBCs of one of the wells detected and labeled. This will serve as the')
disp('basis for the tracking of cells throughout the rest of the video frames.')
disp(' ')

%% Problem 1, Step 7: Evaluate all of the Frames in the Movies

% This section will take an extremely long amount of time to run, therefore
% I suggest you do not run this section, and instead run the next section
% as it already has the data provided for it in the folder. 

% Create new directory to store each annotated frame (the individual frames
% will be combined to produce a video in the next step). 
mkdir 'AnnotatedImages1' 

% Need to find the number of images in the new folder. Here, the height of 
% the variable corresponds to the number of images in the directory folder, 
% but the directory automatically adds 2 extra data points (for storage of 
% relative directories I believe)
video_dir=dir(folder_name);  
length_video=(height(video_dir)-2);

% Set parameters 
positions_final={}; %cell array containing the finalized positions with labels at each frame
initial_frame=firstframe2; %so we can use this as the basis to evaluate the next frame
pixel_change=12;  %this determines how much "change" in cell position is likely/feasible
tot_RBC2=[];

for num_frame=1:length_video
    rbc_des_temp=fullfile(folder_name,[(int2str(num_frame)),'.jpg']);
    frame_read=imread(rbc_des_temp);
    new_frame=imcrop(frame_read,current_frame);
    col_temp=1;
    row_temp=1;
    RBC_position=[];
    for num_row_step=1:v_steps 
        for num_col_step=1:h_steps
            temp_window=new_frame(row_temp:row_temp+(box_width-1),col_temp:col_temp+(box_height-1));  %crop image at window
            temp_features=extractHOGFeatures(temp_window,'CellSize',cell_size,'NumBins',num_bins,'BlockSize',block_size);  %extract HOG features per window
            [prediction,score]=predict(SVM_Classifier1,temp_features);     %determine if window contains RB    
            if prediction==label1    %(RedBloodCell)
                frame_position=[(col_temp+(current_frame(1))),(row_temp+(current_frame(2))),...
                    box_width,box_height];  %this is the [x_leftcorner,y_leftcorner,width,height] value
                RBC_position=[RBC_position; frame_position];
            end
            col_temp=col_temp+stepsize; %move a step to the right until it reaches the edge of the image
        end
        row_temp=row_temp+stepsize; % move a step down the image
        col_temp=1;    %start back at the left side of the image
    end
    % Detect each object only once 
    [RBC_position_final,num_images_final]=IsSameObjectFinal(RBC_position,stepsize); 
    tot_RBC2=[tot_RBC2;RBC_position_final]; 
    % Need to track the cells - I created a user defined function which
    % assigns the windows at each frame to a specific cell, so the cell can
    % be tracked throughout the entire video. 
    [labeled_frame]=CorrectPosition(initial_frame,RBC_position_final,pixel_change);
    
    % Note: labeled_frame is the variable which holds the finalized
    % location values of the cells in this particular frame, along with the
    % number of the label found in the 5th column of the matrix. 
    
    % Place each position/label matrix into a final cell array 
    positions_final1(num_frame,1)={labeled_frame};
    initial_frame=labeled_frame; %so it always analyzes consecutive images
    % Retrieve annotated image and place into new folder
    annotated_frame=insertObjectAnnotation(frame_read,'Rectangle',labeled_frame(:,1:4),labeled_frame(:,5));
    destination=fullfile('AnnotatedImages1',[int2str(num_frame),'-A.jpg']); %create destination with file name ("place holders")
    imwrite(annotated_frame,destination);  %actually write and place the file into the destination
end

%% Problem 1, Step 7 Command Window Display 

disp('Problem 1, Step 7: In this step, the same technique that occurred in step 6 was applied, but')
disp('it was repeated for (almost) every frame in the video. The output of this section is a folder containing')
disp('an image of every frame marked with the location and label of the cells.')
disp(' ')

%% Problem 1, Step 8: Display the video  

% Use VideoWriter to create a video file using the images 
newfolder='AnnotatedImages';
write_vid=VideoWriter('Final_vid.avi');
write_vid.FrameRate=video.FrameRate;  % use the same frame rate as the first video 
video_dir=dir(newfolder);
length_video=(height(video_dir)-2);  %total number of frames
open(write_vid)
for aframe=1:length_video
    rbc_des_temp=fullfile(newfolder,[(int2str(aframe)),'-A.jpg']);
    frame_read=imread(rbc_des_temp);
    writeVideo(write_vid,frame_read) %place each image into the video file 
end
close(write_vid)

% Play the video 
implay('Final_vid.avi')

disp('Problem 1, Step 8: Here, I have combined all of the annotated frames into a .avi video file, which should be')
disp('displayed in a separate window. Overall, I believe that the cell detection using feature extraction/')
disp('classifaction, the function to recognize multiple windows detecting the same object within a single frame')
disp('and the function to track the same cell throughout consecutive images worked well in conjuction with one')
disp('another. But, I definitely think that this system can be improved. There are a few frames (around 5-7) in this well in which the')
disp('window of one of the cells is detected in a position a small distance away from the cell. These inaccuracies could skew')
disp('the results found in the second problem of the project. Nonetheless, this specific well works relatively well with')
disp('the model I have created, I find that this level of accuracy is not found in other wells using this model. I think that,')
disp('by providing a larger, more detailed training set, the feature extractor would be more accurate, and therefore')
disp('the classifier would find less false positives/negatives. While the extractor/classifier is currently pretty accurate (I') 
disp('would guess around 80-90% accuracy) it could definitely still use improvement. Also, I think that tracking function needs to be much')
disp('more finely tuned, because in other wells where the cells move greater distances, the tracking labels become much more confused.') 
disp('Adding a feature extractor to the trackig function might be a good option for this as well. All in all, the model I created')
disp('performs accurately in this well, but making this program more flexible is the future goal.')
disp(' ')

%% Problem 2, Step 1: Retrieve the Position Vectors for Each Cell 

% Adjust the cell array containing all of the position coordinates into
% a matrix which organizes the coordinates by which cell they belong to.
% Then, we can call on specific columns of the matrix to retrieve the
% positions of each cell through all of the frames. 

% Need to load the cell array containing all of the positions of the cells 
% in the well in all of the frames. This cell array was created while creating 
% the annotated images in step 7. 
load('positions_final.mat');

% Parameters: 
cell_positions=[];
numcells=height(cell_names);  
length_pos=height(positions_final);

% Loop through all of the cell positions and organize the positions of the
% same cells in one column. 
for image=1:length_pos
    full=positions_final{image,1};  
    cell_col=full(:,5);
    full_piece=full(:,[1:4]);
    full_fix=[cell_col,full_piece]; %place the cell numbers at the beginning of the positions matrix
    full_finish=sort(full_fix);     %sort the matrix so that the cell numbers are in order
    length_full=height(full_fix);  
    piece_tot=[];
    for indiv=1:length_full
        piece_full=full_finish(indiv,:);    %extract an individual vector from the frame's position matrix
        piece=convert(piece_full,current_frame); %use the function created below to change the coordinates
        piece_tot=[piece_tot,piece];   %place it into a vector, which is organized by cell number
    end
    cell_positions=[cell_positions;piece_tot];   %compile all of the vectors into a large matrix which contains
end                                         %every cell's location sequentiall through the entire video

% Note: cell_positions holds the distances of the x and y values of each
% cell at each frame. The first 2 columns correspond to the x & y values
% for the first cell, the second 2 columns correspond to the x & y values
% for the second cell, etc. These are all organized in time as they are
% presented in the video. 

disp('Problem 2, Step 1: The cell array containing all of the cell position information')
disp('has been transformed into an organized matrix which separates each cell from one another.')
disp('Also, the coordinates have been changed so that [x,y] describe the displacement from')
disp('the centers of the cells to the walls of the well. It should be noted that the centers of')
disp('the cell are determined by the position of the window created around them, so this might cause')
disp('a bit of inaccuracy.')
disp(' ')

%% Problem 2, Step 2: Determine Displacements Between Each Frame 

% Need to find the difference between x and y positions in each cell in all
% of the frames. 

cell_displacements=[];
new_length=length_pos-1;   %need to subtract one from the length to avoid error 
for row=1:new_length 
    first=cell_positions(row,:);  % cell positions in the first frame
    second=cell_positions(row+1,:);  % cell positions in the consecutie frame 
    displacement=abs(second-first);   %take the absolute value of the subtraction!
    cell_displacements=[cell_displacements;displacement];
end

disp('Problem 2, Step 2: The displacements between x and y coordinates in consecutive')
disp('frames for each cell have been calculated.')
disp(' ')

%% Problem 3, Step 3: Plot the Histograms of the Differences 

% I will display a histogram of every cell, with the displacements in both
% x and y integrated. 

disp('Problem 2, Step 3: In a separate window the histogram for cell 1 should be plotted.')
disp('To view the next cells histogram, press any button.')
disp(' ')

[dis_rows,dis_cols]=size(cell_displacements);  
count=-1;
fignum=5;
while count~=(dis_cols-1)  %end loop once it has worked through all of the cell data 
    count=count+2;  % +2 because plotting both x and y for the cells
    fignum=fignum+1;  %figure number so each histogram can be saved separately
    figure(fignum)
    histogram([cell_displacements(:,count),cell_displacements(:,(count+1))])  %plot the histogram
    new_str=sprintf('Displacement In Cell %.f Position Between Frames in Both X and Y Directions',(fignum-5));
    title(new_str)
    xlabel('Displacement Between Frames')
    ylabel('Number of Occurences')
    pause
end

disp('Now that each of the cells histograms (which include both the x and y data displacement)')
disp('have been plotted, you can view all of the cells histograms plotted on top of one another by')
disp('pressing any key.')
disp(' ')
pause

% Now, I have the displacements of all cells added to one histogram 
count=0;
while count~=(dis_cols-1)
    count=count+1;
    figure(11)
    histogram(cell_displacements(:,count))
    title('Displacement In Cell Position Between Frames in Both X and Y Directions')
    xlabel('Displacement Between Frames')
    ylabel('Number of Occurences')
    hold on
end

disp('We can see that the displacement with the greatest number of occurrences is 0 (by a very')
disp('large margin). Also, the displacement rarely exceeds over 5 points (and if it does, it is')
disp('likely an error in the placement which I had described before). If I had not converted all displacements')
disp('to a positiven number, this would likely very closely resembles a guassian distribution, which')
disp('characterizes random motion. This is exactly what would we would expect of particles')
disp('which follow Brownian motion. It can also be seen that all of the cells do follow the same sort')
disp('of "curve", so uniformly and individually express random placement (which is also as we would expect).')
disp('All in all, we can see that RBCs in a fluid medium exhibit Brownian motion, with a diffusion coefficient')
disp('which is described by the Einstein-Stokes equation.')

%% Functions 

% Function to change coordinates found in Problem 2, Step 1: 

% Need to create a function which converts the [cell_num, xtopleft,
% ytopleft, x_width, y_height] coordinates to [xcenter, ycenter] coordinates, 
% and finds their relative locations with respect to the walls of the well 
% they are in (wall coordinates are [xtopleft, ytopleft, x_width, y_height].
% This function returns the distance from the coordinate to the wall it is 
% nearest to (in both x and y). 
function [new_coor]=convert(coor_vec,frame_vec)
x=(coor_vec(2)+(coor_vec(4)/2));  %center of cell 
y=(coor_vec(3)+(coor_vec(5)/2));  %center of cell
x_well=(frame_vec(1)+(frame_vec(3)/2));  %center of well
y_well=(frame_vec(2)+(frame_vec(4)/2));  %center of well
if x<=x_well    %"if the cell is closest to the left wall (or right in the middle)"
    x_diff=x-frame_vec(1);   %x coordinate - left hand side 
else   %"if x is closest to the right wall"
    x_diff=(frame_vec(1)+frame_vec(3))-x;   %right hand side - x coordinate
end 
if y<=y_well    %"if x is closest to the left wall (or right in the middle)"
    y_diff=y-frame_vec(2);   %x coordinate - left hand side 
else   %"if x is closest to the right wall"
    y_diff=(frame_vec(2)+frame_vec(4))-y;   %right hand side - x coordinate
end
new_coor=[x_diff,y_diff];
end








    

