function qualityResult = check_image_quality(imagePath)
% CHECK_IMAGE_QUALITY Evaluates retinal image gradability according to clinical criteria
% NETRADRISHTI - Smart India Hackathon 2026 (SIH26038)
%
% Criteria:
%   1. Focus / Blur: 2D Laplacian variance on green channel
%   2. Illumination: Mean intensity & overexposure/glare ratio
%   3. Field of View: Active retinal area percentage
%
% Output:
%   qualityResult - Struct with status, scores, reason, and recommendation

rawImg = imread(imagePath);
if size(rawImg, 3) == 1
    rawImg = repmat(rawImg, [1, 1, 3]);
end

greenChan = single(rawImg(:, :, 2));
lum = single(rgb2gray(rawImg));
[h, w] = size(lum);
totalPixels = h * w;

% 1. Retinal Mask & Field of View
retinaMask = lum > 18;
retinaPixels = sum(retinaMask(:));
fovPercentage = (retinaPixels / totalPixels) * 100;

% 2. Illumination Analysis
if retinaPixels > 0
    meanIllum = mean(lum(retinaMask));
    overexposedCount = sum(lum(retinaMask) > 240);
    glareRatio = overexposedCount / retinaPixels;
else
    meanIllum = mean(lum(:));
    glareRatio = 1.0;
end

% 3. Blur Estimation via Laplacian Variance
lapKernel = [0,  1, 0; ...
             1, -4, 1; ...
             0,  1, 0];
lapImg = conv2(greenChan, lapKernel, 'same');
blurScore = var(lapImg(:));

% Thresholds (consistent with Python Quality Gate)
BLUR_THRESHOLD = 50.0;
MIN_ILLUM = 40.0;
MAX_ILLUM = 225.0;
MIN_FOV = 45.0;

reasons = {};

if blurScore < BLUR_THRESHOLD
    reasons{end+1} = 'Image too blurry';
end
if meanIllum < MIN_ILLUM
    reasons{end+1} = 'Poor illumination (Underexposed)';
elseif meanIllum > MAX_ILLUM || glareRatio > 0.28
    reasons{end+1} = 'Poor illumination (Severe flash glare / Overexposed)';
end
if fovPercentage < MIN_FOV
    reasons{end+1} = 'Insufficient field of view';
end

qualityResult = struct();
qualityResult.blurScore = blurScore;
qualityResult.illuminationScore = meanIllum;
qualityResult.fovScore = fovPercentage;

if ~isempty(reasons)
    qualityResult.status = 'UNGRADABLE IMAGE';
    qualityResult.usable = false;
    qualityResult.reason = reasons{1};
    qualityResult.recommendation = 'Please recapture the fundus image.';
    qualityResult.qualityScore = max(5.0, min(48.0, (blurScore/BLUR_THRESHOLD)*20.0 + fovPercentage*0.25));
else
    qualityResult.status = 'IMAGE ACCEPTED';
    qualityResult.usable = true;
    qualityResult.reason = 'None';
    qualityResult.recommendation = 'Image quality verified. Ready for AI screening.';
    qualityResult.qualityScore = min(99.0, max(65.0, (blurScore/350)*40 + (1 - abs(meanIllum-120)/100)*35 + (fovPercentage/75)*25));
end

fprintf('[NETRADRISHTI Quality Gate] Status: %s | Score: %.1f | Blur: %.1f | Illum: %.1f\n', ...
    qualityResult.status, qualityResult.qualityScore, qualityResult.blurScore, qualityResult.illuminationScore);

end
