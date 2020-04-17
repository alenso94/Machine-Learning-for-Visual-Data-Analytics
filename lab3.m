% % information
%facial age estimation
%regression method: linear regression

% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 

N = length(ytest);%number of labels
Ne=0;%to calculate cs_number
for i = 1:N
    diff = abs(ytest(i,1) - yhat_test(i,1));
    absTestErr = absTestErr + diff;
    if diff <= err_level
       Ne = Ne + 1;
    end
end
absTestErr = absTestErr/N;
cs_number = (Ne/N)*100;

%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
 cs_nums = zeros(15,1);
 for j = 1:15
    Ne=0;
    for i = 1:N
        diff = abs(ytest(i,1) - yhat_test(i,1));
        if diff <= j
            Ne = Ne + 1;
        end
    end
    cs_nums(j,1) = (Ne/N)*100;
 end
 figure;
 plot(cs_nums(:,1));%plot the cumulative score
 grid on
 title('CS value against the cumulative error level');
 xlabel('Error level')
 ylabel('Cumulative Score')
 xlim([0 16])
 ylim([0 100]) 
 
% %% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.
%Partial least square Regression 
[XL,YL,XS,YS,beta,PCTVAR] = plsregress(xtrain,ytrain,5);%5 is chosen as it gives considerably less error and relatively high cumulative score compared to the number of components

 ytest_PLS = [ones(size(xtest,1),1) xtest]*beta;
 Ne=0;
 absTestErr_PLS = 0;
 for i = 1:N
     diff = abs(ytest(i,1) - ytest_PLS(i,1));
     absTestErr_PLS = absTestErr_PLS + diff;
     if diff <= err_level
        Ne = Ne + 1;
     end
 end
absTestErr_PLS = absTestErr_PLS/N;
cs_number_PLS = (Ne/N) * 100;
%Regression tree
%Mdl = fitrtree(xtrain,ytrain);
Mdl = fitrtree(xtrain,ytrain,'MaxNumSplits',11);%11 is chosen as it gives least error and considerable CS
ytest_RT = predict(Mdl,xtest);
Ne=0;
absTestErr_RT = 0;
for i = 1:N
     diff = abs(ytest(i,1) - ytest_RT(i,1));
     absTestErr_RT = absTestErr_RT + diff;
     if diff <= err_level
        Ne = Ne + 1;
     end
end
absTestErr_RT = absTestErr_RT/N;
cs_number_RT = (Ne/N) * 100;
% %% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox
% Please note that here Libsvm is used: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
% compile it when neccesary
run('libsvm-3.24/matlab/make')
addpath 'libsvm-3.24\matlab'
model = svmtrain(ytrain, xtrain, '-s 3 -t 0');%training the SVR model
ytest_SVR = svmpredict(ytest, xtest, model);
 Ne=0;
 absTestErr_SVR = 0;
 for i = 1:N
     diff = abs(ytest(i,1) - ytest_SVR(i,1));
     absTestErr_SVR = absTestErr_SVR + diff;
     if diff <= err_level
        Ne = Ne + 1;
     end
 end
absTestErr_SVR = absTestErr_SVR/N;
cs_number_SVR = (Ne/N) * 100;
