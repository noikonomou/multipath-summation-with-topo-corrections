function output = c3_hyperbola_fitting(real_im)
% This file presents the c3_algorithm for detecting and fitting hypobola
% form Ground Penetrating Radar (GPR) B-scan images.

% real_im is a greyscale GPR image
% output is a structure storing the (x, y) coordinates of all fitted hyperbolae

% Plese cite the following article if you are using this code in your work:
%   Dou Q, Wei L, Magee D, Cohn A. 2017. 
%   Real-Time Hyperbola Recognition and Fitting in GPR Data. 
%   IEEE Transactions on Geoscience and Remote Sensing. 51-62 55.1 

% This function was copied from
% https://github.com/lweileeds/hyperbola_recognition/tree/main/c3_algorithms
% and it was modified to exctract information for Ground Penetrating Radar) GPR velocities

close all

real_im = double(real_im);

tic
% Step 1: region clustering using C3 alorithm


[xx, yy, xxx, yyy] = column_connection_clustering_v2(real_im,0.25, 2, 2, 0.001);%rustaq


x = -ones(3,length(xx));

% Step 2: fitting hyperbola
% Extract the ncc value of detected clusters w.r.t. a predefined hyperbola
for i = 1:size(yyy,1)
    [dy1, dy2, ~, ~] = ncc_values_v2(xxx{i,1},yyy{i,1});
    x(1,i) = dy1;
    x(2,i) = dy2;
end

% Change the values of y coordinates into negative
for i = 1:size(yy,1)
    yy{i,1} = -yy{i,1};
    yyy{i,1} = -yyy{i,1};
end

% Load the trained Neural Network hyperbola variables of v and w
load('trained_vectors.mat') 
y = -ones(size(x));
vv = v*x;
y(1:2,:) = 2./(1 + exp(-vv)) - 1;
out_put = 2./(1 + exp(-w*y)) - 1;

subplot(2,3,6);
imagesc(real_im);
colormap gray(256);
title('Final results')

output = {};
ind_plus = find(out_put > 0);
if ~isempty(ind_plus)
    ind_plus = find(out_put > 0);
    xxx = xxx(ind_plus);
    xx = xx(ind_plus);
    yyy = yyy(ind_plus);
    yy = yy(ind_plus);
    out_v = cluster_cleaning(xx, yy);
    out_v = logical(out_v);
    xxx = xxx(out_v);
    xx = xx(out_v);
    yyy = yyy(out_v);
    yy = yy(out_v);
clear xyv1 c
count1=0;
    for i=1:size(xx,1)
        for j=1:length(xx{i,1})
            im_regions(-yy{i,1}(j), xx{i,1}(j))=255;
        end
        subplot(2, 3, 5);
        imagesc(im_regions); 
        colormap gray(256);axis off;
        title('Hyperbola-like regions');
        % Fitting
        subplot(2, 3, 6); hold on;
        [a, b, xc, yc,a_ini, b_ini, xc_ini, yc_ini] = G_N_hyperbola_fitting_v2(xxx{i,1}, yyy{i,1}, 5);
        if ~isreal(a) || ~isreal(b) || a<0 || b<0 || a==inf || b==inf || isnan(a) || isnan(b)
            a = a_ini;
            b = b_ini;
            yc = yc_ini;
            xc = xc_ini;
        end
        if a < 1
            continue
        end
        h_interval = max(abs(xc-xxx{i,1}(1)),abs(xc-xxx{i,1}(end)));
        xs = (xc-h_interval-2):(xc+h_interval+2);
        ys = -a*sqrt(1+(xs-xc).^2/b^2)+yc;
        output(i).xs = xs;
        output(i).ys = ys;
        output(i).xc=xc;
        output(i).yc=yc;
        output(i).a=a;
        output(i).b=b;
        aa=output(i).a+abs(output(i).yc);



 % START of addition by Nikos Economou------------------------
 
 % New dt and dx, based on the image png
ratex=30/length(real_im(1,:));ratey=48.8759/length(real_im(:,1));% for rustaq
tt0=min(-ys).*ratey;
tt02=-ys(end);
xx0=xc.*ratex;
xx02=xs(end).*ratex;

% store the above
output(i).yv=tt0;
output(i).xv=xx0;

%FIT line x^2-t^2 over the half hyperbola
starthy=floor(length(xs)/2)+1;
xhy=((xs(starthy:end)-xs(starthy)).*ratex).^2;
yhy=(ys(starthy:end).*ratey).^2;
p = polyfit(xhy,yhy,1);
output(i).vhy=(4/p(1)).^(0.5);

 % END of addition by Nikos Economou-----------------------------
        
        plot(xs, -ys, 'b-', 'linewidth', 2);
        drawnow

%         if isempty(hyperbolae(i).v);continue ;
%     else;
%         count1=count1+1;
%         xyv(count1,3)=vv;xyv(count1,2)=tt0;xyv(count1,3)=vv;
%         end
        


    end
end 


