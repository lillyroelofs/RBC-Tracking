%% Develop Training Images for Classifier

% As a warning: The paths for the images are saved under my personal
% directory information, so running this file may not work? I am not sure,
% but it is also not necessary to run this file since the image folder
% ("PrimaryTrainingImages") is already placed into the folder. 

% I used the image labeler app to create labels and locations of both red
% blood cells and non-red blood cells in individual frames of the video.
% This method was used to provide a distinction for the classifier, so a
% machine learning approach could be taken. I placed labels only in wells 
% in which all 4 sides were not visible. I did this because those wells 
% cannot be used for this project, and therefore they provide 
% outside data to be trained on. It is important not to train algorithms on 
% the same data you are testing to avoid bias, so that was the purpose 
% of me doing this.

% The labeleling sessions are saved in 2 .mat files, "imageLabelingSession1.mat"
% and "imageLabelingSession2.mat". This was the result of my laptop crashing and 
% damaging one of my files, so it  became complicated and making 2 
% separate sessions was the easiest way to do it. Anyways, if you would like 
% to view this session and see the specific cells I labeled in each image, 
% go to the "Apps" tab and click "Image Labeler", and load this file. Since I am only
% trying to detect if a cell is present (regardless of which phase it is
% in), I have all of the cells labeled under one ROI ('RBC_Cell') label, 
% and all non-cell/background images labeled under 'Not_RBC'.

% Now, need to load the data into the workspace

image_label_data=load("gTruth1.mat"); 
gTruth=image_label_data.gTruth; %this is the ground truth data that I exported from the image labeling session

% Create a table containing the location of the box as well as the label
% asociated with it 
[im_datastore,boxlabel_datastore]=objectDetectorTrainingData(gTruth);

% Note: boxlabel_datastore.LabelData is the name of the array that stores
% the pixel and label information. Each entry in the first column contains 
% each box's location information [x_center, y_center, x_length, y_height] 
% per image, and the second column contains the label information per box.

% Now, I would like to save these labels as separate cropped images in their 
% own subfolder, with the name of the subfolder as the label for the images.

label1=gTruth.LabelDefinitions{1,1}{1,1};
label2=gTruth.LabelDefinitions{2,1}{1,1};
mkdir('PrimaryTrainingImages',label1) 
mkdir('PrimaryTrainingImages',label2)
cd 'PrimaryTrainingImages'   %must move to this directory to save the images

% Need to find the number of pictures in training data 
num_trainimages=height(boxlabel_datastore.LabelData); 

% Need to loop through each of these images as well as the boxes in them,
% and create new images of each of these (while placed in folder
% corresponding to label)
for pic_num=1:num_trainimages
    num_entries=height(boxlabel_datastore.LabelData{pic_num,1});  %find number of boxes in each image
    for num_box=1:num_entries
        og_name=im_datastore.Files{pic_num,1}; %returns a string of the path to the original image 
        og_image=imread(og_name); %imports the image
        box_crop=imcrop(og_image,[boxlabel_datastore.LabelData{pic_num,1}(num_box,1),...
            boxlabel_datastore.LabelData{pic_num,1}(num_box,2),boxlabel_datastore.LabelData{pic_num,1}(num_box,3),...
            boxlabel_datastore.LabelData{pic_num,1}(num_box,4)]);  %crops the images as described in the imagestore         
        [path,filename,ext]=fileparts(og_name);
        if (boxlabel_datastore.LabelData{pic_num,2}(num_box,1))==label1     %only for RedBloodCell labels
            destination1=fullfile(label1,[filename,'-',int2str(num_box),'.jpg']); %create destination with file name ("place holders")
            imwrite(box_crop,destination1,'Quality',100);  %actually write and place the file into the destination 
        else
            destination2=fullfile(label2,[filename,'-',int2str(num_box),'.jpg']); %only for non-RedBloodCell labels
            imwrite(box_crop,destination2,'Quality',100); 
        end
    end 
end

cd ..   %used to navigate back to the SuperProject directory



% Need to repeat the whole process for the second file 
image_label_data2=load("gTruth2.mat"); 
gTruth2=image_label_data2.gTruth; %this is the ground truth data that I exported from the image labeling session

% Create a table containing the location of the box as well as the label
% asociated with it (this refers to boxLabelDatastore)
[im_datastore2,boxlabel_datastore2]=objectDetectorTrainingData(gTruth2);

% Now, I would like to save these labels as separate cropped images in their 
% own subfolder, with the name of the subfolder as the label for the images.

% Need to find the number of pictures in training data 
num_trainimages2=height(boxlabel_datastore2.LabelData); 

% Need to loop through each of these images as well as the boxes in them,
% and create new images of each of these (while placed in folder
% corresponding to label)

cd 'PrimaryTrainingImages'

for pic_num=1:num_trainimages2
    num_entries=height(boxlabel_datastore2.LabelData{pic_num,1});  %find number of boxes in each image
    for num_box=1:num_entries
        og_name=im_datastore2.Files{pic_num,1}; %returns a string of the path to the original image 
        og_image=imread(og_name); %imports the image
        box_crop=imcrop(og_image,[boxlabel_datastore2.LabelData{pic_num,1}(num_box,1),...
            boxlabel_datastore2.LabelData{pic_num,1}(num_box,2),boxlabel_datastore2.LabelData{pic_num,1}(num_box,3),...
            boxlabel_datastore2.LabelData{pic_num,1}(num_box,4)]);  %crops the images as described in the imagestore        
        [path,filename,ext]=fileparts(og_name);
        if (boxlabel_datastore2.LabelData{pic_num,2}(num_box,1))==label1
            destination1=fullfile(label1,[filename,'-N',int2str(num_box),'.jpg']); %create destination with file name ("place holders")
            imwrite(box_crop,destination1,'Quality',100);  %actually write and place the file into the destination 
        else
            destination2=fullfile(label2,[filename,'-N',int2str(num_box),'.jpg']); %create destination with file name ("place holders")
            imwrite(box_crop,destination2,'Quality',100);  %actually write and place the file into the destination 
        end
    end
end

cd ..   %used to navigate back to the SuperProject directory


% Now, need to resize these images to make them all the same size 

% Use imageDatastore as an efficient way of obtaining image files. Also,
% using this method the folder name is treated as the label for each image,
% which will be helpful. 
training_images=imageDatastore('PrimaryTrainingImages','IncludeSubfolders',1,'LabelSource','foldernames');

% First, we need to find the total number of images in our training set 
tot_num_images=numel(training_images.Files);

% Determine the average size of all of the images
pixel_len=0;
pixel_height=0;
for image_num=1:tot_num_images
    temp_img=readimage(training_images,image_num);
    temp_size=size(temp_img);
    pixel_len=pixel_len+temp_size(1);
    pixel_height=pixel_height+temp_size(2);
end

% Find the average size, then add a multiplier in order to slightly raise
% the size of the images. I found that this helped with image detection as
% it allowed for larger cells to be detected. 
avg_len=round((pixel_len/tot_num_images)*1.12);    
avg_height=round((pixel_height/tot_num_images)*1.11);

% Now, resize the images and place them into the same directory we made
% above (to avoid cluttering the folder)
cd 'PrimaryTrainingImages' 

for pic_num=1:num_trainimages
    for image_num=1:tot_num_images
        og_name=training_images.Files{image_num,1};
        temp_img=readimage(training_images,image_num);
        resized_image=imresize(temp_img,[avg_len,avg_height]);
        [path,filename,ext]=fileparts(og_name);  %this allows for the file to be named correctly in the new folder
        if training_images.Labels(image_num,1)==label1
            destination1=fullfile(label1,[filename,'.jpg']); %create destination with file name ("place holders")
            imwrite(resized_image,destination1,'Quality',100);  %actually write and place the file into the destination 
        else
            destination2=fullfile(label2,[filename,'.jpg']); %create destination with file name ("place holders")
            imwrite(resized_image,destination2,'Quality',100);  %actually write and place the file into the destination 
        end
    end
end

cd ..   %return back to main directory 
