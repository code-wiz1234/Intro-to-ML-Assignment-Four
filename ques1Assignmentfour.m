clc; clear; close all;

rng(42); 
N_train = 1000;
N_test = 10000;
sigma = 1;
r_neg = 2;
r_pos = 4;


theta_train = 2*pi*rand(N_train,1) - pi;
x_neg_train = r_neg * [cos(theta_train(1:N_train/2)), sin(theta_train(1:N_train/2))] + sigma*randn(N_train/2,2);
x_pos_train = r_pos * [cos(theta_train(N_train/2+1:end)), sin(theta_train(N_train/2+1:end))] + sigma*randn(N_train/2,2);
X_train = [x_neg_train; x_pos_train];
y_train = [-ones(N_train/2,1); ones(N_train/2,1)];


theta_test = 2*pi*rand(N_test,1) - pi;
x_neg_test = r_neg * [cos(theta_test(1:N_test/2)), sin(theta_test(1:N_test/2))] + sigma*randn(N_test/2,2);
x_pos_test = r_pos * [cos(theta_test(N_test/2+1:end)), sin(theta_test(N_test/2+1:end))] + sigma*randn(N_test/2,2);
X_test = [x_neg_test; x_pos_test];
y_test = [-ones(N_test/2,1); ones(N_test/2,1)];

C_vals = logspace(-1, 2, 4);
gamma_vals = logspace(-2, 1, 4);
bestCV = 0;

fprintf('\n--- SVM Cross-Validation ---\n');
for C = C_vals
    for gamma = gamma_vals
        t = templateSVM('KernelFunction','rbf','KernelScale',1/sqrt(2*gamma),'BoxConstraint',C);
        CVSVM = fitcsvm(X_train, y_train, 'KernelFunction','rbf', ...
            'KernelScale',1/sqrt(2*gamma), 'BoxConstraint',C, ...
            'KFold',5);
        acc = 1 - kfoldLoss(CVSVM);
        fprintf('C=%.2f, gamma=%.2f, CV Accuracy=%.2f%%\n', C, gamma, acc*100);
        if acc > bestCV
            bestCV = acc;
            bestSVM = fitcsvm(X_train, y_train, 'KernelFunction','rbf', ...
                'KernelScale',1/sqrt(2*gamma), 'BoxConstraint',C);
            bestC = C;
            bestGamma = gamma;
        end
    end
end


y_pred_svm = predict(bestSVM, X_test);
svm_error = mean(y_pred_svm ~= y_test);
fprintf('\nBest SVM hyperparameters: C=%.2f, gamma=%.2f, CV Accuracy=%.2f%%\n', bestC, bestGamma, bestCV*100);
fprintf('SVM Test Error: %.4f\n', svm_error);

hidden_sizes = [2 5 10 20 50];
best_mlp_error = inf;
fprintf('\n--- MLP Cross-Validation ---\n');

for h = hidden_sizes
    net = patternnet(h);
    net.trainFcn = 'trainscg';
    net.performFcn = 'crossentropy';
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'softmax';
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.0;
    net.trainParam.showWindow = false;

    y_train_01 = (y_train + 1)/2;
    y_train_onehot = [1 - y_train_01'; y_train_01'];

    [net, tr] = train(net, X_train', y_train_onehot);
    valInd = tr.valInd;
    valPred = net(X_train(valInd,:)');
    [~, valClass] = max(valPred);
    valClass = 2*(valClass' - 1.5);
    valError = mean(valClass ~= y_train(valInd));

    fprintf('Hidden Neurons=%d, CV Accuracy=%.2f%%\n', h, (1-valError)*100);

    if valError < best_mlp_error
        best_mlp_error = valError;
        best_net = net;
        best_h = h;
    end
end

y_test_pred = best_net(X_test');
[~, y_pred_mlp] = max(y_test_pred);
y_pred_mlp = 2*(y_pred_mlp' - 1.5);
mlp_error = mean(y_pred_mlp ~= y_test);

fprintf('\nBest MLP hidden neurons: %d, CV Accuracy=%.2f%%\n', best_h, (1-best_mlp_error)*100);
fprintf('MLP Test Error: %.4f\n', mlp_error);

[x1Grid,x2Grid] = meshgrid(linspace(-6,6,300), linspace(-6,6,300));
XGrid = [x1Grid(:), x2Grid(:)];

[~, score_svm] = predict(bestSVM, XGrid);
Z_svm = reshape(score_svm(:,2), size(x1Grid));

mlp_scores = best_net(XGrid');
[~, mlp_pred] = max(mlp_scores);
Z_mlp = reshape(2*(mlp_pred - 1.5), size(x1Grid,1), size(x1Grid,2));

figure;
subplot(1,2,1);
gscatter(X_test(:,1), X_test(:,2), y_test, 'rb', '..');
hold on;
contour(x1Grid, x2Grid, Z_svm, [0 0], 'k', 'LineWidth', 2);
title(sprintf('SVM Decision Boundary (Error = %.2f%%)', svm_error*100));
axis equal;

subplot(1,2,2);
gscatter(X_test(:,1), X_test(:,2), y_test, 'rb', '..');
hold on;
contour(x1Grid, x2Grid, Z_mlp, [0 0], 'k', 'LineWidth', 2);
title(sprintf('MLP Decision Boundary (Error = %.2f%%)', mlp_error*100));
axis equal;