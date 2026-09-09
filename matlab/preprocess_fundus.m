function [processedImg, retinaMask] = preprocess_fundus(imagePath, targetSize)
% PREPROCESS_FUNDUS Prepares retinal fundus image for quality gating and AI inference
% NETRADRISHTI - Smart India Hackathon 2026 (SIH26038)
%
% Inputs:
%   imagePath  - Path to input fundus image (.png, .jpg)
%   targetSize - 2-element vector [height, width] (default: [224, 224])
%
% Outputs:
%   processedImg - Normalized RGB tensor [224 x 224 x 3] ready for inference
%   retinaMask   - Binary logical mask of retinal aperture

if nargin < 2
    targetSize = [224, 224];
end

% 1. Read input image
rawImg = imread(imagePath);
if size(rawImg, 3) == 1
    rawImg = repmat(rawImg, [1, 1, 3]);
end

% 2. Extract luminance and green channel
% In ophthalmology, the green channel provides maximum contrast for retinal microvasculature
greenChan = rawImg(:, :, 2);
lum = rgb2gray(rawImg);

% 3. Extract circular retinal mask (aperture)
retinaMask = lum > 18;

% 4. Contrast-Limited Adaptive Histogram Equalization (CLAHE) on Green Channel
enhancedGreen = adapthisteq(greenChan, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');

% 5. Composite enhanced image
enhancedRGB = rawImg;
enhancedRGB(:, :, 2) = enhancedGreen;

% 6. Resize to standard CNN input dimension
processedImg = imresize(enhancedRGB, targetSize);
retinaMask = imresize(retinaMask, targetSize, 'nearest');

% Normalize to [0, 1] single precision
processedImg = single(processedImg) / 255.0;

end
