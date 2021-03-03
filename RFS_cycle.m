
close all
load data_train
load ranks

sequence = [6,7,8,1,13,4,3];

c = 1.4142; g = 1;

c = 2; g = 1.4142;

train_textures = data_train;

label_train = train_textures(:,end);
label_train(find(label_train==0)) = -1;	


sequence = ranks;

AUC_max = [];


for i = 1:length(sequence)
    
    seq = sequence(i);
     
    auc_j = zeros(1,length(sequence));  
  
    auc_max = plot_roc(label_train,train_textures(:,seq),'-v 5',c,g);;
   
for k = 1:length(sequence)-1
    
    for j = 1:length(sequence)
        if ~any(seq==sequence(j))   
          auc_j(j) = plot_roc(label_train,train_textures(:,[seq,sequence(j)]),'-v 5',c,g);
        end
    end
    
    [~,index] = find(auc_j==max(auc_j));
    
    auc_max = [auc_max ,max(auc_j)];
    
    seq = [seq,sequence(index(1))];
    
    auc_j = zeros(1,length(sequence));  
end

    AUC_max = [AUC_max,max(auc_max)]

%            if max(auc_j)>auc
%                auc = [auc,auc_j]
%                seq = [seq,sequence]; 
%             end
%          
end

[~,ind] = find(AUC_max==max(AUC_max))



%% -----------------------------------
i = 13;

seq = sequence(i);

auc_j = zeros(1,length(sequence));

auc_max = plot_roc(label_train,train_textures(:,seq),'-v 5',c,g);;

for k = 1:length(sequence)-1
    
    for j = 1:length(sequence)
        if ~any(seq==sequence(j))   
          auc_j(j) = plot_roc(label_train,train_textures(:,[seq,sequence(j)]),'-v 5',c,g);
        end
    end
    
    [~,index] = find(auc_j==max(auc_j));
    
    auc_max = [auc_max ,max(auc_j)];
    
    seq = [seq,sequence(index(1))];
    
    auc_j = zeros(1,length(sequence));  
end


plot([1:length(sequence)],auc_max,'sr','MarkerFaceColor','r')
hold on
plot([1:length(sequence)],auc_max,'r-','LineWidth',1)
hold on

% plot([1:16],auc_max(10)*ones(1,25),'b--',10*ones(1,2),[0.4,1],'b--')

xlabel('Features Dimension');
ylabel('AUC');
grid on

% plot([1:25],auc_up:auc_down ,'k.')

% plot([1:25],0.01*acc,'sb','MarkerFaceColor','b')
% hold on 
% plot([1:25],0.01*acc,'b-')


