% Written by Dr. Nikos Economou-Technical University of Crete, Chania, Greece
% Was created and run with MatLab 2022b
%%This script demonstrates how to obtain Figure 5 and Figure 6 (with constant velocity topography correction) from the paper below:
%___________________________________________________________________________________
% Economou, N., Douglas, K., Hessein, M., Al Hooti, K., Pizzimenti, S., Khan, M., Al Jahwari, N., 
% Al Shaqsi, B., Al Abri, S., Al Hinaai, A., Al Kharusi, A., Al Alawi, M., Al Rawahi, R., 
% Al Ismaili, S., 2026, A methodology for imaging Early Bronze Age structures buried in mounds at Al-Tikha, Oman, 
% Archaeological Prospection, accepted.
%______________________________________________________________________________________
% These figures are showing the proposed methodology for multipath
% summation and topography correction.


% 

% %Before running it you must:

% 1. Download the C3 algorithm from 
% https://github.com/lweileeds/hyperbola_recognition

% 2. Place the downloaded package from step 1 above in the directory
% "hyperbola_recognition-main" in the directory Matlab and according to the instructions.

% 3. Replace the function "hyperbolae = c3_hyperbola_fitting(real_im)" with the function "hyperbolae = c3_hyperbola_fitting1(real_im)" 
% from here. The latter contains changes so that is works for GPR velocities.
% 
% 4. Make sure that the path to the main directory "hyperbola_recognition-main" is set to Matlab.
%
% 5. Place the present script in the main directory "hyperbola_recognition-main" and run it.
%



% The present script uses the functions mentioned above to run a multipath
% summation methodology combined with ML for difractions idendification and topograpy corrections 
% which is descibed in the paper 1 below.
% The multipath summation methodology is described in detail in papers 2 and 3 below.  

%1. Economou, N., Douglas, K., Hessein, M., Al Hooti, K., Pizzimenti, S., Khan, M., Al Jahwari, N., 
% Al Shaqsi, B., Al Abri, S., Al Hinaai, A., Al Kharusi, A., Al Alawi, M., Al Rawahi, R., 
% Al Ismaili, S., 2026, A methodology for imaging Early Bronze Age structures buried in mounds at Al-Tikha, Oman, 
% Archaeological Prospection, accepted.

%2. Economou, N., S. Nasir, S. Al-Abri, B. Al Shaqsi and H. Hamdan. 2024. 
% "AI Aided GPR Data Multipath Summation Using x-t Stacking Weights," 47th International 
% Conference on Telecommunications and Signal Processing (TSP), Prague, Czech Republic: 285-288.

%3.	Economou, N., Vafidis, A., Bano, M., Hamdan, H., and Ortega-Ramirez, J., 2020, 
% Ground-penetrating radar data diffraction focusing without a velocity model, 
% Geophysics, 85, no. 3, 1-12. IF: 2.391.

% For any questions you can contact me at noikonomou@tuc.gr



% 
%
 clear all;close all
 NoLp=18; %Line number
 k=1;% column in the data array
 load (['Line', num2str(NoLp),'.mat']); % Load GPR section
 load (['diffr_mat', num2str(NoLp),'.mat']);% Load GPR section with relfections extracted to enhance diffractions 


 DATA=Line18{1,1};
 dt=Line18{4,1}(2);
 dx=Line18{6,1}(2);
 t=Line18{4,1};t=t(:);
 x=Line18{6,1};
  

 real_im = imread(['rustaq_diffr' num2str(NoLp),'.png']); % Read the GPR section's png image (The C3 algorithm starts on images)
 real_im=sum(real_im(:,:,:),3);% transform the image to dataset for the C3 algorithm

% Below we will detect and fit hyperbolae using the proposed C3 algorithm, with the
% modified "function c3_hyperbola_fitting1", which is fixed to estimate
% kinematic characteristics of the hyperbolas and extract the coordinates of 
% the hyperbolas and the migration velocities.

% The ouput we wil use in the vector "hyperbola.vhy", i.e. the migration
% velocities matrix.

% If you do not want to use the C3 algorithm, you can create a matrix in an array with
% the name hyperbolae.vhy consisting of 3 columns:
% x-coordinate of the hyperbola, y-coordinate of the hyperbola, velocity value

%FIGURE 1
% Pots also a multi-figure plot of the ML procedure as it is made by the
% creators of the C3 algorithm
hyperbolae = c3_hyperbola_fitting1(real_im); % Running the c3 algorthm
h=gcf;set(h,'Position',[10,40,1200,400])
     hgexport(gcf, ['Fig1_Image_from_C3_algorithm',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')

%Make it a comment of you provide your own hyperbolae.vhy matrix

%FIGURE 2
%GPR section plot
 figure('Position',[10 40 1200 400], 'color','white');;pcolor(Line18{6,k}, Line18{4,k},Line18{1,k});shading interp;colormap(bone);
 ylabel('Time (ns)'); xlabel('Distance (m)');title(['GPR Section',num2str(NoLp)])
 set(gca,'ydir','reverse','xaxislocation','bottom','yaxislocation','left','layer','top','linewidth',2,'tickDir','out','box','on')
 ax=gca;ax.FontSize=16;
 clim([min(min(DATA))./3 max(max(DATA./3))])
 hgexport(gcf, ['Fig2_GPR Section',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
% END FIGURE 2


%FIGURE 3
%GPR section with diffractions enhanced plot, superimposing the detected hyperbolas
figure('Position',[10 40 1200 400], 'color','white');pcolor(real_im);shading interp;colormap(bone)
 ylabel('Image samples'); xlabel('Image samples');title(['Diffractions Section Line',num2str(NoLp)]);
 set(gca,'ydir','reverse','xaxislocation','top','yaxislocation','left','layer','top','linewidth',2,'tickDir','out','box','on')

 hold on
for i=1:length(hyperbolae)
plot(hyperbolae(i).xs, -hyperbolae(i).ys, '-b','linewidth', 2);
plot(hyperbolae(i).xc, min(-hyperbolae(i).ys), '*')
plot(hyperbolae(i).xc, (-hyperbolae(i).yc), '*')

end
ax=gca;
ax.FontSize=16;

hgexport(gcf, ['Fig3_Diffractions Section',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png');
%END FIGURE 3


% Set the velocities min and max values to be involved in the section. 
minvel=0.075;
maxvel=0.115;

count1=0;
for i=1:length(hyperbolae);

 if isempty(hyperbolae(i).vhy) || (isnan(hyperbolae(i).vhy)) ; continue ;% for rustaq
   
 
else;
     
    count1=count1+1;
    xyv(count1,1)=floor(hyperbolae(i).xv./dx);
    xyv(count1,2)=floor(hyperbolae(i).yv./dt);
    if (hyperbolae(i).vhy<minvel) 
        xyv(count1,3)=minvel;
    elseif (hyperbolae(i).vhy>maxvel)
        xyv(count1,3)=maxvel;
    else
    xyv(count1,3)=hyperbolae(i).vhy;;
    end
    
end;
end

eval(['hyperbolae',int2str(NoLp),'=hyperbolae;']) ;
eval(['xyv',int2str(NoLp),'=xyv;']) ;

% Create the 3 x,y,v vectors

xx=xyv(:,1);
yy=xyv(:,2);
vv=xyv(:,3);

%Interpolation__________________________________________!!!!!!!!!!!


xmax=length(DATA(1,:));
ymax=length(DATA(:,1));
xmin=1;stepx=1;
ymin=1;stepy=1;
[xq,yq] = meshgrid(xmin:stepx:xmax,ymin:stepy:ymax);

% 1 way of interpolation
% F = scatteredInterpolant(x,y,v,'natural','nearest');
% %F.Method = 'natural';
% vq1 = F(xq,yq);

% 2nd way of interpolation
% griddata (BETTER!!!)
vq1 = griddata(xx,yy,vv,xq,yq,'v4'); % V4 better!!!


% end of Interpolation%_____________________________!!!!!!!!!!!!!


% %__________________________________________________4444444444444444444444444
% % normalze from minv to maxv: vq1norm=(vq1-minvq1)/(maxvq1-minvq1).*(newmaxvq1-newminvq1)+newminvq1
 vq11=vq1;
minvelold=min(min(vq1));
maxvelold=max(max(vq1));
minvelnew=0.075;maxvelnew=0.115;
 vq1norm=minvel+((vq1-minvelold)./(maxvelold-minvelold)).*(maxvelnew-minvelnew);
  vq1=vq1norm;
% end of normalize_______________________________4444444444444444444444444


%FIGURE 4
 figure('Position',[10 40 1200 400], 'color','white');;pcolor(Line18{6,k}, Line18{4,k},(vq1));shading interp;%hold on;%plot(x,y,'ok') ;...
    colorbar;colormap(jet);clim([0.06 0.14]);ylabel('Time (ns)'); xlabel('Distance (m)');title([['Diffraction velocity model Line',num2str(NoLp)]]);
set(gca,'ydir','reverse','xaxislocation','bottom','yaxislocation','left','layer','top','linewidth',2,'tickDir','out','box','on')
colorbar;h = colorbar;ylabel(h, 'Velocity (m/ns)')
ax=gca;ax.FontSize=16;
hgexport(gcf, ['Fig4_Diffraction velocity model Line',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
%END of FIGURE 4


%FIGURE 5
sz=55;figure('Position',[10 40 1200 400], 'color','white');;scatter(xx,yy,sz,vv,'filled');colorbar
set(gca,'ydir','reverse','xaxislocation','top','yaxislocation','left','layer','top','linewidth',2,'tickDir','out','box','on')
 ylabel('Image samples'); xlabel('Image samples');title([['Diffraction apices vs velocities Line',num2str(NoLp)]]);
 colormap(jet);clim([0.06 0.14]); colorbar;h = colorbar;ylabel(h, 'Velocity (m/ns)')
xlim([xmin xmax])
ylim([ymin ymax])
ax=gca;ax.FontSize=16;
hgexport(gcf, ['Fig5_Diffraction apices vs velocities Line',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
%END of FIGURE 5





%%%%%%%%%%%%%%%%%%%___________Multipath2D_______________________AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


%%%%%%%%%%%%%%%%%%%Make 2D filters for multipath summation

disp(['Making the 2D filters...'])
vels=0.01:0.01:0.3; % create the vels big-range vector

count1=0;
velsforproc=[minvel-0.02:0.01:maxvel+0.02];% We open the velocity range 
for vii=velsforproc

count1=count1+1;
yg = gaussmf(((1:length(vels))), [length(vels)/14 vii*100]);%figure;plot(vels,yg,'o-') % Make a gaussian with maximum value at vii

%Create the 2D filters
vjkfilt=vq1.*0;
for j=1:length(vq1(:,1));
    for k=1:length(vq1(1,:));
vjk=vq1(j,k);

        [c d]=min(abs(vels-vjk));
        vjkapprox=vels(d);
        
        vjkweight=yg(d);
        vjkfilt(j,k)=vjkweight;
    end
end
% store them in a 3D matrix
AA=vjkfilt;
Anorm=(AA-min(min(AA)))./(max(max(AA))-min(min(AA))).*(1-0+0);
vjkfilt_3D(:,:,count1)=Anorm;
end

%smooth 2D and normalize to 0 to 1
[sivjrow sivjcol]=size(vjkfilt_3D(:,:,count1));
N=[80 80];
  leftproduct = spdiags(ones(sivjrow,2*N(1)+1),(-N(1):N(1)),sivjrow,sivjrow);
rightproduct = spdiags(ones(sivjcol,2*N(2)+1),(-N(2):N(2)),sivjcol,sivjcol); 

for cc1=1:count1
    matin=vjkfilt_3D(:,:,cc1);A = isnan(matin);nrmlize(A) = 0;
nrmlize = leftproduct*(~A)*rightproduct;matin(A) = NaN;
matin=(leftproduct*matin*rightproduct);
matin=matin./nrmlize;
AA=matin;
Anorm=(AA-min(min(AA)))./(max(max(AA))-min(min(AA))).*(1-0+0);%Norm from 0 to 1
vjkfilt_3Dsm(:,:,cc1)=Anorm;
end
% End of smooth 2D and normalize to 0 to 1


% End of the creation of 2D filters



% % Plot an example of the 2D filters for velocity velsforproc(numv)_22222222222222222222222222222
 numv=6;
 A=vjkfilt_3D(:,:,numv);
 B=vjkfilt_3Dsm(:,:,numv);


%FIGURE 6
% Plot an example of the 2D filters for velocity velsforproc(numv)
figure('Position',[10 40 1200 400], 'color','white');pcolor(Line18{6,1}, Line18{4,1},A);shading interp;%hold on;plot(x,y,'ok') ;...
 colorbar;colormap(jet)%caxis([min(min(vq1)) max(max(vq1))]);
set(gca,'ydir','reverse','xaxislocation','bottom','yaxislocation','left','layer','top','linewidth',2,'tickDir','out','box','on')
ylabel('Time (ns)'); xlabel('Distance (m)');title(['2D-weights for mig-vel ', num2str(velsforproc(numv)), 'm/ns'])
     ax=gca;ax.FontSize=16;
     hgexport(gcf, ['Fig6_2D-weights for mig-vel',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
%END of FIGURE 6


%FIGURE 7
% Plot an example of the 2D filters for velocity velsforproc(numv)
figure('Position',[10 40 1200 400], 'color','white');pcolor(Line18{6,1}, Line18{4,1},B);shading interp;%hold on;plot(x,y,'ok') ;...
 colorbar;colormap(jet)%caxis([min(min(vq1)) max(max(vq1))]);
set(gca,'ydir','reverse','xaxislocation','bottom','yaxislocation','left','layer','top','linewidth',2,'tickDir','out','box','on')
ylabel('Time (ns)'); xlabel('Distance (m)');title(['Smoothed 2D-weights for mig-vel ', num2str(velsforproc(numv)), 'm/ns'])
     ax=gca;ax.FontSize=16;
     hgexport(gcf, ['Fig7_Smoothed 2D-weights for mig-vel',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
%END of FIGURE 7


% END of plot an example of the 2D filters for velocity velsforproc(numv)_222222222222222222222222222

%%%%%%%%%%%%%%%%%%%%%END of make 2D filters for multipath summation




%%%%%%%%%%%%%%%%%%%%% 2D Weighted stacking 

disp([num2str(length(velsforproc)),' migrations and stacking...'])
siIPDd=size(DATA);
    sumtemp=zeros(siIPDd(1), siIPDd(2));
    sumtempabs=sumtemp;
    countv=0;
    for v1=velsforproc;
countv=countv+1


%%%%%%%%%                          Migration
Din=DATA;v=v1;
S=KzOperator(Din,v,dx,dt); %Dr. Nikos Spanoudakis-GPRPro-See at the end of the script
[nt,nx]=size(Din);p=fft2(Din);S=imresize(S,[size(Din)]);
P={};PP={};new=[];P{1}=p.*S;PP{1}=sum(P{1});new(1,:)=PP{1};
    for j=2:nt
    P{j}=P{j-1}.*S;PP{j}=sum(P{j});new(j,:)=PP{j};P{j-1}=[];PP{j-1}=[];
    end
dmig=real(ifft(new,[],2)); 
%figure;imagesc(dmig)
%%%%%%%%%%%             END OF     Migration

migs{1,countv}=dmig;
migs{2,countv}=v1;

% Equation of multipath summation-equation 2 in paper 2 above.
sumtemp=sumtemp+dmig.*vjkfilt_3Dsm(:,:,countv);%weighted stacking
sumtempdia=sumtemp./sum(vjkfilt_3Dsm,3);% normalize to the sum of weights-Final focused result

    end
 %%%%%%%%%%%%%%%%%%%%% End of 2D Weighted stacking 

 
% FIGURE 8
figure('Position',[10 40 1200 400], 'color','white');;pcolor(Line18{6,1}', Line18{4,1},sumtempdia);shading interp;colormap(bone);%clim([-100 100])
 set(gca,'ydir','reverse','xaxislocation','bottom','yaxislocation','left','layer','top','linewidth',2,'tickDir','out','box','on')
      ylabel('Time (ns)'); xlabel('Distance (m)');title(['Section focused ',num2str(NoLp)]);
ax=gca;ax.FontSize=16;clim([min(min(sumtempdia))./3 max(max(sumtempdia./3))])
hgexport(gcf, ['Fig9_Section_focused',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
% END of FIGURE 8

% This figure can be directly compared with Fig 2 to assess focusing

%%%%%%%%%%%%%%%%%%%___________END of Multipath2D_______________________AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

 


% Below topogaphic corrections are implemented, based on the paper best
% results (Fig. 6b). If one wants to implement topographic migration then the
% algorithm from 
% ^^^^^^^^Wunderlich, T. 2024. "MultichannelGPR: A MATLAB tool for Ground Penetrating Radar data processing". Journal of Open Source Software 9,no. 102: 6767.^^^^^^^^^^
% can be used. It can be found in github: https://github.com/tinawunderlich/MultichannelGPR




% %TOPO Corrections

v1=0.11; %Correction velocity;
zmax=270.96; % max elevation of the area of interest
load elevL18; % Topographic info: elevation over study line
dxelev=0.5; % dx of topographic info
y=0:dxelev:((length(elevL18))-1)*dxelev;
yi = interp1(0:dxelev:x(end)-dxelev,elevL18,x);yi=yi(:); % Interpolate elevation info over study line x vector



% Topographic correction for velocity model-------------------------
data=vq1norm;
[ns , ntr] = size(data);
datacorr = zeros(ns,ntr);
hw = waitbar(0,'Topographic Correction ...');
for trace = 1:ntr
    selev = yi(trace); tcorr = 2*(-selev + zmax )/(v1);% Time correction estimation
        % For every time for all traces 
    ts = zeros(ns,1);
    for itime=1:ns
        ts(itime) = (itime-1)*dt - tcorr;
    end
% Interpolate new data 
    tr1 = interp1(t,data(:,trace),ts,'cubic',0); 
% Load corrected data array
    datacorr(:,trace) = tr1;
    waitbar(trace/ntr);
end
close(hw)
datacorr(datacorr==0)=NaN; % for white sky

% Normalize to initial vel range again 
minvelold=min(min(datacorr));
maxvelold=max(max(datacorr));
minvelnew=minvel;maxvelnew=maxvel;
datacorr=minvel+((datacorr-minvelold)./(maxvelold-minvelold)).*(maxvelnew-minvelnew);
velcorr=datacorr;
depth=t.*0.5.*v1;

% FIGURE 9
figure('Position',[10 40 1200 400], 'color','white');pcolor(x,zmax-(depth),velcorr);shading interp;colormap(jet)%colormap(bone)
axis([-0   30  268.3 271.0]) 
ax=gca; ax.FontSize=16;colorbar;h = colorbar;ylabel(h, 'Velocity (m/ns)');
xlabel('Distance (m)');ylabel('Elevation (m) (m)');title(['Section vel model topocorr Line',num2str(NoLp)]);
clim([0.06 0.14])
hgexport(gcf, ['Fig9_Section_vel_topocorr',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
% END FIGURE 9


% END of Topographic correction for velocity model--------------------------



%Topographic correction for focused GPR section---------------------
data=sumtempdia;
[ns , ntr] = size(data);
datacorr = zeros(ns,ntr);
hw = waitbar(0,'Topographic Correction ...');
for trace = 1:ntr
    selev = yi(trace); tcorr = 2*(-selev + zmax )/(v1);% Time correction estimation
        % For every time for all traces 
    ts = zeros(ns,1);
    for itime=1:ns
        ts(itime) = (itime-1)*dt - tcorr;
    end
% Interpolate new data 
    tr1 = interp1(t,data(:,trace),ts,'cubic',0); 
% Load corrected data array
    datacorr(:,trace) = tr1;
    waitbar(trace/ntr);
end
close(hw)
datacorr(datacorr==0)=NaN; % for white sky

focusedcorr=datacorr;
depth=t.*0.5.*v1;

% FIGURE 10
figure('Position',[10 40 1200 400], 'color','white');pcolor(x,zmax-(depth),focusedcorr);shading interp;colormap(bone)
axis([-0   30  268.3 271.0]) 
ax=gca; ax.FontSize=16;
xlabel('Distance (m)');ylabel('Elevation (m) (m)');title(['Section topocorr Line',num2str(NoLp)]);
clim([min(min(focusedcorr))/3.5 max(max(focusedcorr))/3.5])
hgexport(gcf, ['Fig10_Section_vel_topocorr',num2str(NoLp),'.png'], hgexport('factorystyle'), 'Format', 'png')
% End of FIGURE 10

%END of topographic correction for focused GPR section--------------------


%END of TOPO corrections








% Function used for Kz operator to be used for constant velocity migration

function [S] = KzOperator(Din,v,dx,dt);
%
%  IN    v: velocity in m/nsec
%        dx: step size in m
%        dz: depth discretization in m
%        dt: sampling interval in nsec
%
%  OUT   S: kz operator 


%Implemented by Dr. Nikos Spanoudakis
%
%S. N. Spanoudakis and A. Vafidis, "GPR-PRO: A MATLAB module for GPR data processing," 
% Proceedings of the XIII Internarional Conference on Ground Penetrating Radar, Lecce, Italy, 2010, 
% pp. 1-5, doi: 10.1109/ICGPR.2010.5550131.

 [nt,nx]=size(Din);
 dkx=2*pi/(nx*dx);
 dw=2*pi/(nt*dt);
 dz=v*dt/2;
 v=v/2;
 kx=[0:dkx:(nx/2)*dkx -((nx/2)*dkx-dkx):dkx:-dkx];
 w =[0:dw:(nt/2)*dw -((nt/2)*dw-dw):dw:-dw];
 
 index=find(w == 0);
 w(index)=1E-18;
 
 [kx w] = meshgrid(kx,w);
 
 ARG = 1.0-(kx.^2*v^2)./(w.^2);
 index=find(ARG < 0.);
 ARG(index) = 0.;

 S = zeros(nt,nx);
 S = exp(i*w/v.*sqrt(ARG)*dz);
 S(index) = 0.0;
end