clc
clear;
peopleNum = 68;
trainNum =25;
testNum = 24;
m=64;
n=64;
load('D:\wps\DPLS\demo_Yale_2DPLS\dataset\PIE_64×64.mat')
      
%%
        Xtr = NewTrain_DAT'/255;
        Xte = NewTest_DAT'/255;
        [n1,m1] = size(Xtr);
        M = eye(n1)-ones(n1,n1)/n1;
        N = eye(m1)-ones(m1,m1)/m1;
        Xtrain = M*Xtr*N;
        
        meanXtrain =mean(Xtr,1);
        meanXtest = mean(Xte,1);
        Xtrain = bsxfun(@minus,Xtr, meanXtrain);
        Xtest =bsxfun(@minus,Xte, meanXtest);
        n_class=68;
          trainLabel=[];
        Ytr = zeros(size(Xtr,1),n_class);
        for k=1:68
            temp = ones(1,trainNum)*k;
            trainLabel = [trainLabel,temp];
            Ytr(trainNum*(k-1)+1:trainNum*k,k) = ones(trainNum,1);
        end
         meanYtrain = mean(Ytr,1);
         Ytrain = bsxfun(@minus,Ytr,meanYtrain);
        clear temp
        testLabel=testlabels;
    %%
 jj =0;
for ncomp = 36
 jj = jj+1;
%%
      
num_layers = 4;  
Xtrain_new = Xtrain;  
Xtest_new = Xtest;  
window_size = 3;  
Xtrain_smoothed = smoothdata(Xtrain, 'gaussian', window_size);  
Xtest_smoothed = smoothdata(Xtest, 'gaussian', window_size);  
for layer = 1:num_layers  
    beta  =[5,10,2,12];
    % 计算当前层的PLS投影矩阵  
    W_layer = PLSRGGr(Xtrain_new, Ytrain, ncomp, beta(layer));
      
    % 计算训练集和测试集的得分  
    T_layer = Xtrain_new * W_layer;  
    Tt_layer = Xtest_new * W_layer;  
      Xtrain_new1=Xtrain_new;
      Xtest_new1=Xtest_new;
    % 将当前层的得分与平滑后的数据（如果需要）组合  
    if layer < 2  
        Xtrain_new = [T_layer,Xtrain_smoothed];  
        Xtest_new = [Tt_layer,Xtest_smoothed];  
    else
        Xtrain_new = [T_layer, T_prev_layer];
        Xtest_new = [Tt_layer, Tt_prev_layer];
    end
% 保存当前层的得分作为上一层的得分，以便下一轮循环使用
    T_prev_layer = T_layer;
    Tt_prev_layer = Tt_layer;
    
end 

Xloadings = Xtrain_new1'*T_layer/(T_layer'*T_layer);
Yloadings = Ytrain'*T_layer/(T_layer'*T_layer);
B = W_layer*Yloadings';
Yhat_tr =bsxfun(@plus,Xtrain_new1*B, meanYtrain);
Yhat_te =bsxfun(@plus,Xtest_new1*B,meanYtrain);

[~,Ytrain_label]=max(Yhat_tr');
[~,Ytest_label]=max(Yhat_te');
err_train=1- length(find(trainLabel-Ytrain_label==0))/size(trainLabel,2);
err_test=1- length(find(testLabel-Ytest_label==0))/size(testLabel,2)
end

 % 计算混淆矩阵
num_classes = 68; % 你的类别总数
conf_matrix = confusionmat(testlabels,Ytest_label, 'Order', 1:num_classes);

% 初始化各项
TP = zeros(1, num_classes);
TN = zeros(1, num_classes);
FP = zeros(1, num_classes);
FN = zeros(1, num_classes);

% 计算每个类别的混淆矩阵项
for i = 1:num_classes
    TP(i) = conf_matrix(i, i);
    FN(i) = sum(conf_matrix(i, :)) - TP(i);
    FP(i) = sum(conf_matrix(:, i)) - TP(i);
    TN(i) = sum(conf_matrix(:)) - TP(i) - FN(i) - FP(i);
end

% 计算每个类别的 Recall、Precision、F-measure 和 G-mean
recall = TP ./ (TP + FN);

% 处理 Precision 和 F-measure 中的分母为零情况
precision_denominator_zero = (TP + FP) == 0;
precision = TP ./ (TP + FP);
precision(precision_denominator_zero) = 0;

% 处理 F-measure 中的分母为零情况
f_measure_denominator_zero = (precision + recall) == 0;
f_measure = 2 * precision .* recall ./ (precision + recall);
f_measure(f_measure_denominator_zero) = 0;

% 计算每个类别的 TNR（True Negative Rate）
tnr = TN ./ (TN + FP);

% 计算每个类别的 G-mean
g_mean = sqrt(recall .* tnr);

% 计算宏平均（Macro-average）
macro_recall = mean(recall);
macro_precision = mean(precision);
macro_f_measure = mean(f_measure);
macro_g_mean = mean(g_mean);

% 显示结果
disp(['Recall (Macro-average): ', num2str(macro_recall)]);
disp(['Precision (Macro-average): ', num2str(macro_precision)]);
disp(['F-measure (Macro-average): ', num2str(macro_f_measure)]);
disp(['G-mean (Macro-average): ', num2str(macro_g_mean)]);