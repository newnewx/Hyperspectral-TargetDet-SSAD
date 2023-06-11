clear
load sandiego
clear seg
seg = reshape(double(imread('./sandiego1_200.png'))/255,length(map),1);
data=hyperNormalize(data);

[L, N] = size(data);
seg=reshape(seg,sqrt(N),sqrt(N));
data1=reshape(data',sqrt(N),sqrt(N),L);
% d=make_d(data1,map);
d=make_d(data1,seg);

clear auc fpr FPR rst thre TPR lambda weight data1 L N
function [d] = make_d(data,seg)
%% Ôìd
    [a,b,c] = size(data);
    N=a*b;
    sortseg=sort(reshape(seg,N,1),'descend');
    len = length(find(sort(sortseg)~=0));
    seg_tmp=sortseg(1:ceil(len*0.3));  %È¡Ç°30%ÏñËØ
    [row,col,bands] = size(data);
    d_i = zeros(1,1,bands);
    image_tmp = double(data);
    n = 0;
    for i = 1:1:row
        for j = 1:1:col
            if (seg(i,j) >= min(seg_tmp(:))) 
                    d_i = d_i+image_tmp(i,j,:);
                    n=n+1;
            end
        end
    end
    d = d_i./(n);
    d = reshape(d,1,bands);
    d=d';
%     plot(d,'r','Linewidth',2);
%     axis([0 200 1000 3200]); 
%     saveas(gcf,'d','jpg');
    da=reshape(data,N,c);
    data=da';
    clear bands col d_i da i image_tmp j n row BW level seg_tmp sortseg len data1 a b c N
%     seg=double(seg);
end