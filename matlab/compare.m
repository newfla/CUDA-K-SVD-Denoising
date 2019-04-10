%/MISC VAR
inputFolder = '/home/flavio/Progetti/Tesi/img/input';
outputFolder = '/home/flavio/Progetti/Tesi/img/output';
fileName = '/dora_';
fileExt = '.jpg';

%/ITERATE ON SAME NOISE_RANGE
for i=0.1:+0.05:0.5
    
    inputFilePath = strcat(inputFolder, fileName, num2str(i), fileExt);
    outputFilePath = strcat(outputFolder, fileName, num2str(i), fileExt);
    
    REF = imread(inputFilePath);
    MOD = imread(outputFilePath);
    
    [peaksnr, snr] = psnr(MOD, REF);
    [ssimval, ssimmap] = ssim(MOD, REF);
    
    fprintf(strcat('\nFile name: ',fileName, num2str(i)));
    
    fprintf('\nThe Peak-SNR value is %0.4f', peaksnr);
    fprintf('\nThe SNR value is %0.4f \n', snr);
    fprintf('The SSIM value is %0.4f.\n',ssimval);
    
    fprintf('-----------------------------------------------\n');
    
    
end