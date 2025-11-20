%% GMM-based Image Segmentation 

clear; close all; clc;


imagePath = 'C:\Users\Owner\Downloads\102061.jpg'; 
img = imread(imagePath);
img = im2double(img); 
[rows, cols, ~] = size(img);


[R_grid, C_grid] = ndgrid(1:rows, 1:cols);
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
features = [R_grid(:), C_grid(:), R(:), G(:), B(:)];

for i = 1:5
    f = features(:,i);
    features(:,i) = (f - min(f)) / (max(f) - min(f));
end

Kvals = 2:8;      
numFolds = 5;    
avgLogL = zeros(size(Kvals));

fprintf('Running cross-validation for K=%d to %d...\n', Kvals(1), Kvals(end));

for k = 1:length(Kvals)
    K = Kvals(k);
    cv = cvpartition(size(features,1), 'KFold', numFolds);
    L = zeros(numFolds,1);

    for fold = 1:numFolds
        trainIdx = training(cv, fold);
        valIdx   = test(cv, fold);

       
        gm = fitgmdist(features(trainIdx,:), K, ...
                       'RegularizationValue', 1e-5, ...
                       'Options', statset('MaxIter',500));

        L(fold) = mean(log(pdf(gm, features(valIdx,:))));
    end

    avgLogL(k) = mean(L);
    fprintf('K=%d, mean val log-likelihood=%.4f\n', K, avgLogL(k));
end

[~, idx] = max(avgLogL);
bestK = Kvals(idx);
fprintf('Best number of components K=%d\n', bestK);

bestGM = fitgmdist(features, bestK, ...
                   'RegularizationValue', 1e-5, ...
                   'Options', statset('MaxIter',500));

clusterLabels = cluster(bestGM, features);
labelImage = reshape(clusterLabels, rows, cols);

minLabel = min(labelImage(:));
maxLabel = max(labelImage(:));
segmented = (labelImage - minLabel) / (maxLabel - minLabel);

figure;
subplot(1,2,1); imshow(img); title('Original Image');
subplot(1,2,2); imshow(segmented); title('GMM Segmentation (Grayscale)');

