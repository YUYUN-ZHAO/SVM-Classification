%
% ------------------------------------------------------------------------
%
%    The function of this code is:
%    1. PCA algorithm
%    2. Relief algorithm
%    3. Cross-validation and Grid search method 
%    4. Establish SVM model and predict in training sets and test sets
%
% ------------------------------------------------------------------------

%
%  Attention:  
%  ----------
%        This code is based on the 'libsvm' toolkit developed by Professor 
%   Lin of Taiwan University.
%        If you want to run this code, please install the 'libsvm' toolbox firstly.
%   


%% 
close all;
clear; 

%% Import data
load data_train;
load data_test;
load ranks;

label_train = data_train(:,end);
label_test = data_test(:,end);

%%  PCA
%{
% [pc,score,latent,~] = pca( data_train_normalized(:,1:end-1));
% rank = cumsum(latent)./sum(latent);
% 
% trainMatrix = pc(:,1:11);
% data = [ mapminmax( (data_train_normalized(:,1:end-1)*trainMatrix)' ,-1,1);data_train_normalized(:,end)' ]';
% 
% features = data(:, 1:end-1); 
%
% textures_train = mapminmax(score(:,1:11),0,1);
%}

%% Relief algorithm

x = data_train(:,1:end-1);
y = data_train(:,end);

[ranks,weights] = relieff(x,y,10);

subplot(1,2,1);
bar(weights(ranks))
xlabel('Predictor rank')
ylabel('Predictor importance weight')

subplot(1,2,2);
bar(weights(ranks))
xlabel('Predictor rank (Top 50 features)')
ylabel('Predictor importance weight')
axis([0 50 0 0.4])

save('ranks','ranks');


%% 

sequence = [4792,4567,7182,10148,12341,2983,68,6928,2898,5030,1509,9953,3411,512,10582,2287];

textures_train = data_train(:,sequence);
textures_test = data_test(:,sequence);

%%  Cross-validation and Grid search method 

% [bestauc,bestc,bestg] = auc_SVMcg(label_train,textures_train,  0 ,8 , -4, 4 ,10 , 0.5,0.5, 0.25);
% 
% [bestacc,bestc_acc,bestg_acc] = SVMcg(label_train,textures_train,  -4 ,8 , -8, 6 ,10, 0.5,0.5, 0.25);

%% 训练模型后进行预测

c = 1.4142; g = 1;

cmd = [' -c ',num2str( c ),' -g ',num2str( g ),' -b 1'];
model = svmtrain(label_train,textures_train,cmd);
[label_pre,mse,decision_values] = svmpredict(label_test,textures_test,model,'-b 1');
% --------------
[label_pre_1,mse_1,decision_values_1] = svmpredict(label_train,textures_train,model,'-b 1');
AUC_train = AUC(label_train,decision_values_1(:,1))
% --------------
%% 计算AUC
AUC_test = AUC(label_test,decision_values(:,1))  
ACC_test = sum(label_test == label_pre)/length(label_pre)
ACC_train =  sum(label_train == label_pre_1)/length(label_pre_1)
Value = [AUC_train;AUC_test;ACC_train;ACC_test];
Name = ['AUC_train';'AUC_test_';'ACC_train';'ACC_test_']
table( Value, Name) 

%% --------------------------------------------------------------

[test_x,test_y, T  ] = perfcurve(label_test',decision_values(:,1)','1');
%      'NBoot',1000,'XVals',[0:0.05:1]    
[train_x,train_y, T] = perfcurve(label_train',decision_values_1(:,1)','1');
       
% e = errorbar(test_x,test_y(:,1),test_y(:,1)-test_y(:,2),test_y(:,3)-test_y(:,1),'b')
% e.MarkerSize = 0.75;
% e,CapSize = 1;
% % e.Color = 'r';
% xlim([ -0.02,1.02]);
% ylim([ -0.02,1.02]);

h1 = plot(train_x,train_y(:,1),'r','LineWidth',2);
hold on;
h2 = plot(test_x,test_y(:,1),'b','LineWidth',2);
hold on;


line([0 1],[0 1],'LineWidth',2,'Color',[1 1 1]*0.8);

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC curve');
legend([h1,h2],'AUC at Training cohort: 0.873(0.806-0.965)',...
    'AUC at Testing cohort:0.811(0.779-0.856)'...
    ,'BaseLine','Location','SouthEast')





