%/DORA SECTION

CLEAN = imread('barbara160.png');

NOISED = imread('barbara.png');

temp=double(NOISED);

var(temp(:))

%/BW = rgb2gray(RGB);

%/RI = imref2d(size(RGB));

%/Image = figure('Name','B&W','NumberTitle','off');
%/imshow(imnoise(RGB, 'gaussian',0.,.0025), RI);

%/NOISED2 = imnoise(CLEAN, 'gaussian',0.,25);
NOISED2 = temp + 0.0025*randn(size(temp));
ssim(NOISED2, double(NOISED))
imshow(NOISED2);