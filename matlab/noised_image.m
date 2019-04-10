%/MISC VAR
array = {};
j = 1;
fileName = 'dora.jpg';
outputName = 'dora_';
outputFormat = '.jpg';
folder = '/home/flavio/Progetti/Tesi/img/input';

%/LOAD IMAGE
RGB = imread('dora.jpg');

%/CONVERT TO GRAYSCALE
GRAY = rgb2gray(RGB);

%/CREATE A SCALE
RI = imref2d(size(GRAY));

%/GENERATE/DISPLAY/SAVE NOISED IMAGES
for i=0.1:+0.05:0.5
    
    NOISED = imnoise(GRAY, 'gaussian',0.,i);
    
    %/figure('Name',strcat('Variance_',num2str(i)),'NumberTitle','off')
    %/imshow(NOISED)
    
    baseOutput = strcat(outputName, num2str(i), outputFormat);
    fullOutput = fullfile(folder, baseOutput);
    imwrite(NOISED, fullOutput);
    
    
end

