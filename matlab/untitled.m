%/DORA SECTION

RGB_Dora = imread('dora.jpg');
BW_Dora = rgb2gray(RGB_Dora);

RI_Dora = imref2d(size(BW_Dora));



Dora_Image = figure('Name','Dora B&W','NumberTitle','off');
imshow(imnoise(BW_Dora, 'gaussian',0.,.5), RI_Dora);

