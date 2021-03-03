%% --------------------------------------------------------------
% clc;clear;

load data_train
load ranks

train_textures = data_train;

label_train = train_textures(:,end);
label_train(find(label_train==0)) = -1;	

c = 1.4142; g= 1;
c = 2; g = 1.4142;

%% ;； 计算权重第一个对应的特征值的AUC值

AUC_1 =	plot_roc( label_train,train_textures(:,ranks(1)),'-v 10',c,g);

Auc_optimal = AUC_1;

%% 迭代寻找最优特征值组合，种类限制为 n 种 

Optimal_textures = train_textures(:,ranks(1));
sequence = ranks(1);
k = 1;
auc_sequence = AUC_1; 
j = [];

for i = 2:length(ranks)

    current_textures = [Optimal_textures,train_textures(:,ranks(i))];
    Auc_current = plot_roc( label_train ,current_textures,'-v 5',c,g);
    
    auc_sequence = [ auc_sequence, Auc_current ];
    
    if Auc_current > Auc_optimal
        Optimal_textures = current_textures;
        sequence = [sequence,ranks(i)];
        Auc_optimal = Auc_current; 
        j = [j,i];
        if length(sequence)>24 
            break   %特征值数量为25时跳出循环
        end
        
    else 
        current_textures = Optimal_textures;
    end
     
end

plot([1:length(auc_sequence)],auc_sequence,'sr-',  'MarkerFaceColor','r');
hold on
grid on;
% plot([0 14],[0.7886,0.7886],'b--');
% hold on
% plot([9,9],[0.55,0.8],'b--');
plot([0 14],[0.7886,0.7886],'--','Color',[0.6,0.6,0.6],'LineWidth',2);
hold on
plot([9,9],[0.55,0.85],'b--','Color',[0.6,0.6,0.6],'LineWidth',2);

xlabel('Features dimension')
ylabel('AUC');
title('Relief Forward Selection');
ylim([0.55 0.85])

% 
% hold on
% plot(j,0.9*ones(1,19),j,0.97*ones(1,19),'s')
% % 
% axis([0,100,0.88,0.97]);
% xlim([0 100])

 %  [6,7,8,1,13,4,12,11,3,5,2,14,9,10]