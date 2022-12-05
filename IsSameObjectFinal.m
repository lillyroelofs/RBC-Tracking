function [final_position_matrix,count] = IsSameObjectFinal(position_matrix,step)

% This function determines whether multiple windows from the same image are
% actually categorizing the same cell. The purpose of this function is to
% only produce 1 label per cell. 

% Parameters:
x_width=position_matrix(1,3);  %width of window - constant for all images
y_height=position_matrix(1,4);  %height of window - constant for all images 
matrix=position_matrix(:,[1,2]);  %only contains the x and y values 
step2=2*step;    %increased step size 
fin_position={};  %contains all of the x,y corner values of images of the same object
count=0;   %at the end it counts the total number of images
final_position_matrix=[];

%Need to add a [0,0] point at the very last row of the matrix
%in order to create a "stopping" point to terminate the while loop once the
% entire matrix has been evaluated.
matrix=[matrix;[0,0]]; 

% Method: The detector classifies images between specific increments
% (i.e. if the step = 5, the locations will be in incremenents of 5 from each
%  other), and by looking at all of the possible coordinates created by adding +5,
% +10, -5, -10 to both the x and y points of ONE window's coordinates, we can test if
% any other window has a coordinate that is found in the vector containing all 
% of the possibilities. From there we can identify that the two windows are 
% detecting the same cell. To create the "absolute" position of the window 
% surrounding the cell, we calculate the average of both the x and y points 
% and passed those back into the workspace for the image annotations. 

while matrix(1,1)~=0 %This loop terminates once the first coordinate=[0,0] 
                     %(meaning it has already passed through the whole matrix)
    x_vec=matrix(:,1);
    y_vec=matrix(:,2);
    num=1;
    % Determine all of the possible x and y coordinate translations
    x_og=x_vec(num);
    y_og=y_vec(num);
    xup=x_vec(num)+step;
    xdown=x_vec(num)-step;
    yup=y_vec(num)+step;
    ydown=y_vec(num)-step;
    xup2=x_vec(num)+step2;
    xdown2=x_vec(num)-step2;
    yup2=y_vec(num)+step2;
    ydown2=y_vec(num)-step2;
    % Create vectors for both x and y 
    x_vec=[x_og,xup,xdown,xup2,xdown2]; %all possible x values
    y_vec=[y_og,yup,ydown,yup2,ydown2]; %all possible y values
    % Create matrix containing all possible x and y pairs 
    [x_att,y_att]=meshgrid(x_vec,y_vec);  %5x5 matrice with elements repeated by rows
    combo=cat(2,x_att',y_att');       %concatenate the matrices
    combo2=reshape(combo,[],2);   %produces all possible combinations between x and y points
    combination=combo2([2:end],:);   %need to remove original x and y values 
    combo_h=height(combination); %determine the number of possibilities
    % Parameters:
    count=count+1;    %counter increases with every new window being evaluated
    same_image=[1,1];      %created to store all coordinates of the same image
    mat_length=height(matrix);   %the height of the matrix changes every iteration
    for rowc=1:combo_h
        xy_c=combination(rowc,[1,2]); %produces [x,y] for the specific combination pair
        for rowm=1:mat_length
            xy_m=matrix(rowm,[1,2]); %produces [x,y] for each window coordinate
            if xy_c==xy_m   %"if the combination pair is the same as the window coordinate"
               same_image=[same_image;[rowm,1]];   %"then they are finding the same image"
            end
        end     
    end
    im_position=matrix([same_image(:,1)],:);   
    fin_position(count,1)={im_position};   %cell array containing all of the x,y corner values of images of the same object
    matrix([same_image(:,1)],:)=[];   %remove the window coordinates from the original matrix (so we don't produce repeats)
end

% Now need to find the average [x,y] coordinates of a single cell according
% to the several window positions.
for num=1:count
    x_point=floor(mean(fin_position{num}(:,1)));
    y_point=floor(mean(fin_position{num}(:,2)));
    final_position_matrix=[final_position_matrix;[x_point,y_point]]; %contains all of the finalized coordinates
end

% Need to include the x_width and y_height in the final position matrix
x_vec_width=repmat(x_width,count,1);
y_vec_height=repmat(y_height,count,1);
[final_position_matrix]=[final_position_matrix,x_vec_width,y_vec_height];
end

