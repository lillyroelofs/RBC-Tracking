function [sframe_positions] = CorrectPosition(fframe,sframe,pixelchange)

% This function uses the positions of cells between frames to determine
% which cells are actually the same, so it is possible to track individual
% cells throughout a video. 
% Note: 
  % 'fframe' = the coordinates of the first frame, including the labels in the 5th column 
  % 'sframe' = the coordinates of the second frame 
  % 'pixelchange' = number of pixels to add/subtract from the initial coordinate 
  
% Parameters: 
x_width=fframe(1,3);  %width of the window (constant)
y_height=fframe(1,4);   %height of the window (constant)

% Assumption: The very first frame is accurate, and correctly
% identifies the location of all present cells, without adding any extra
% objects.
initial_length=height(fframe);  %correlates to the number of cells 
fframe_len=height(fframe);

% We are going to compare the positions of the current cell to that of the
% previous one, and assign the cell label accordingly.

% Because the feature extractor and classifier are not perfect, there are
% some instances in which a cell is not located and also some where extra
% space (not of cells) are located. Because of this, we must compare image
% positions between consecutive frames and determine which position from
% the first image corresponds to which position in the second image. This
% will take a similar method as the code in the IsSameObjectFinal.m file.

% First, calculate possible shifts in the position of the cell by adjusting
% the x and y corner to every value combination of +pixels and -pixels and 
% calculate all of the possible (within reason) coordinates 
% of the cell between frames. Once a coordinate pair is matched up, we will
% categorize them under a cell label. 

% Number of windows in the second frame 
sframe_len=height(sframe);

% We will be comparing the permutations of the second frame to the actual
% positions values of the first frame and locating if/where matches occur.
% This concept is discussed a bit more thoroughly in the
% "IsSameObjectFinal.m" file.

% Parameters: 
sframe_positions=[];
x_new=sframe(:,1);    
y_new=sframe(:,2);     

%Running through each window in a for loop:
for num=1:sframe_len
    x_temp=x_new(num,1);  %evaluating one x and y combination at a time   
    y_temp=y_new(num,1);
    x_vec=x_temp;   %include the original x and y values in the vectors, 
    y_vec=y_temp;   %as these will contain all of the possible x and y values after pixel addition/subtraction
    for i=1:pixelchange
        x_temp_plus=x_temp+i;
        x_temp_minus=x_temp-i;
        x_vec=[x_vec,x_temp_plus,x_temp_minus];   %row vector of all possible x values
        y_temp_plus=y_temp+i;
        y_temp_minus=y_temp-i;
        y_vec=[y_vec,y_temp_plus,y_temp_minus];   %row vector of all possible y values 
        [x_att,y_att]=meshgrid(x_vec,y_vec);  %5x5 matrice with elements repeated by rows
        combo=cat(2,x_att',y_att');       %combine matrices
        combination=reshape(combo,[],2);   %produces all possible combinations between x and y points
        combo_h=height(combination);     %number of possible coordinates
        for new=1:combo_h
            xy_n=combination(new,[1,2]); %produces [x,y] of one of the combination pairs
            for old=1:fframe_len
                fframe_len=height(fframe);
                xy_o=fframe(old,[1,2]);  %produces the [x,y] of one of the coordinates from the first frame 
                if xy_o==xy_n
                    cell_number=fframe(old,5); %
                    new_position_vec=[x_temp,y_temp,x_width,y_height,cell_number];
                    if isempty(sframe_positions)==1  
                        sframe_positions=[sframe_positions;new_position_vec];
                    else
                        num_vector=sframe_positions(:,5);
                        isthere=find(num_vector==cell_number,1);
                        question=isempty(isthere);
                        num_vector2=sframe_positions(:,1);
                        isthere2=find(num_vector2==new_position_vec(1),1);
                        question2=isempty(isthere2);
                        num_vector3=sframe_positions(:,2);
                        isthere3=find(num_vector3==new_position_vec(2),1);
                        question3=isempty(isthere3);
                        if question==1 && (question2==1 && question3==1)
                           sframe_positions=[sframe_positions;new_position_vec];
                        end
                    end
                end
            end
        end
    end
end

% If there is a window in the second frame which was not used, and the
% second and first frames do not contain the same number of boxes at this point, 
% we need to investigate why. There are 2 possibilities:
% 1. The model identified empty space, and the best option is just to use
% the previous location for the box.
% 2. The model located the cell, but at a distance farther than expected
% (so that the pixels did not cover the translation). 
% Parameters:
final_length=height(sframe_positions);  %length of new positions
initial_length; %length of the first frame 
sframe_len; %length of the second frame initially 

if final_length<sframe_len && final_length<initial_length
% Determine which vectors from the second frame were not used 
     membermat=ismember(sframe,sframe_positions(:,[1:4]),'rows'); 
     leftind=find(membermat==0);
     leftout=sframe(leftind,:); %contains the vectors which were not included 
     % Determine which vectors from first frame had not "matched"
     chosencells=sframe_positions(:,5);
     made_up_vec=(1:initial_length);
     cellmember=ismember(made_up_vec,chosencells);
     cell_ind=find(cellmember==0);
     cellout=fframe(cell_ind,:); %contains the vectors which were not included 
     % Now, are these coordinates from the second image within a larger number of pixels?
     % Go through the same process as above, but with a new pixel value.
     new_pixel=40;
     %Running through each window in a for loop:
     
     %First, parameters
     new_sframe_len=height(leftout);
     cellout_len=height(cellout);
     x_leftout=leftout(:,1);
     y_leftout=leftout(:,2);
     sframe_positions2=sframe_positions;
     for num=1:new_sframe_len
        x_temp=x_leftout(num,1);  %evaluating one x and y combination at a time   
        y_temp=y_leftout(num,1);
        x_vec=x_temp;   %include the original x and y values in the vectors, 
        y_vec=y_temp;   %as these will contain all of the possible x and y values after pixel addition/subtraction
        for i=1:new_pixel
            x_temp_plus=x_temp+i;
            x_temp_minus=x_temp-i;
            x_vec=[x_vec,x_temp_plus,x_temp_minus];   %row vector of all possible x values
            y_temp_plus=y_temp+i;
            y_temp_minus=y_temp-i;
            y_vec=[y_vec,y_temp_plus,y_temp_minus];   %row vector of all possible y values 
            [x_att,y_att]=meshgrid(x_vec,y_vec);  %5x5 matrice with elements repeated by rows
            combo=cat(2,x_att',y_att');       %combine matrices
            combination=reshape(combo,[],2);   %produces all possible combinations between x and y points
            combo_h=height(combination);     %number of possible coordinates
            for new=1:combo_h
                xy_n=combination(new,[1,2]); %produces [x,y] of one of the combination pairs
                for old=1:cellout_len
                    xy_o=cellout(old,[1,2]);  %produces the [x,y] of one of the coordinates from the first frame 
                    if xy_o==xy_n
                        cell_number=cellout(old,5); %
                        new_position_vec=[x_temp,y_temp,x_width,y_height,cell_number];
                        if isempty(sframe_positions2)==1
                            sframe_positions2=[sframe_positions2;new_position_vec];
                        else
                            num_vector=sframe_positions2(:,5);
                            isthere=find(num_vector==cell_number,1);
                            question=isempty(isthere);
                            num_vector2=sframe_positions(:,1);
                            isthere2=find(num_vector2==new_position_vec(1),1);
                            question2=isempty(isthere2);
                            num_vector3=sframe_positions(:,2);
                            isthere3=find(num_vector3==new_position_vec(2),1);
                            question3=isempty(isthere3);
                            if question==1 && (question2==1 && question3==1)
                               sframe_positions2=[sframe_positions2;new_position_vec];
                            end
                        end
                    end
                end
            end
        end
     end
     sframe_positions=sframe_positions2;
end
     
     
% If there is still a cell which was not detected, we will just add the position
% of the cell in the previous frame into this frame. Although I understand
% this is not precise, it is the best way to deal with it. 

if final_length<initial_length
    for num=1:initial_length
        current_num=fframe(num,5);
        total_num=sframe_positions(:,5);
        isfound=find((total_num==current_num),1);
        if isempty(isfound)==1
            sframe_positions=[sframe_positions;fframe(num,:)];
        end
    end
end
    
end

