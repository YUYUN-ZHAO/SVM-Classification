function [bestauc,bestc,bestg] = auc_SVMcg(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%SVMcg cross validation by faruto
%Email:farutoliyang@gmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
%last modified 2009.8.23
%Super Moderator @ www.ilovematlab.cn
%% about the parameters of SVMcg 
if nargin < 10
    accstep = 1.5;
end
if nargin < 8
    accstep = 1.5;
    cstep = 1;
    gstep = 1;
end
if nargin < 7
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
end
if nargin < 6
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
end
if nargin < 5
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
end
if nargin < 4
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
end
if nargin < 3
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
    cmin = -5;
end
%% X:c Y:g cg:acc
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 0;
bestg = 0;
bestauc = 0;
basenum = 2;
for i =1:m
    for j = 1:n
        
            cg(i,j) = plot_roc(train_label,train ,'-v 5', basenum^X(i,j) , basenum^Y(i,j));
            
        if cg(i,j) > bestauc
            bestauc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
        if ( cg(i,j) == bestauc && bestc > basenum^X(i,j) )
            bestauc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end

    end
end
%% to draw the acc with different c & g
[C,h] = contour(X,Y,cg,-0.2:accstep/50:1);   
colormap jet
clabel(C,h,'FontSize',8,'Color','k');
xlabel('log2c','FontSize',10);
ylabel('log2g','FontSize',10);
grid on;