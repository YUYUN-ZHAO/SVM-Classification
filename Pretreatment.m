
%% --------------------------------------------------------------
clc;
clear;

%% 

name = cell(1);
save('name','name')
save('label','label');

label_1 = find(label==1);
label_0 = find(label==0);

save('label_1','label_1');
save('label_0','label_0');

%% label_1 随机划分22个测试集和50个训练集
order_rand_1 = randperm(72);
save('order_rand_1','order_rand_1');

label_test_1 = label_1(order_rand_1(1:22));
label_train_1 = label_1(order_rand_1(23:end));

save('label_test_1','label_test_1');
save('label_train_1','label_train_1');

%% label_1   随机划分48个测试集和150个训练集
order_rand_0 = randperm(198);
save('order_rand_0','order_rand_0');

label_test_0 = label_0(order_rand_0(1:48));
label_train_0 = label_0(order_rand_0(49:end));

save('label_test_0','label_test_0');
save('label_train_0','label_train_0');

%% 保存测试集与训练集
label_test = [ label_test_0;label_test_1 ];
label_train =  [ label_train_0;label_train_1];

save('label_test','label_test');
save('label_train','label_train');

%% 保存特征值

textures = textures(:,[1,2,4,5,6,10:end]);
save('textures','textures');

data_train = [ textures(label_train,:) , label_train ];
data_test = [ textures(label_test,:) , label_test ];

LABEL_train = LABEL(label_train);
LABEL_test = LABEL(label_test);

LABEL_train(LABEL_train==0) = -1;
LABEL_test(LABEL_test==0) = -1;

data_train = [ mapminmax(textures(label_train,:)',0,1); LABEL_train']';
data_test = [ mapminmax(textures(label_test,:)',0,1); LABEL_test']';

save('data_train' , 'data_train');
save('data_test' , 'data_test');

%% 取训练集与测试集数据
name_test = name(label_test);
label_TEXT = label(label_test);

name_train = name(label_train);
label_TRAIN = label(label_train);
