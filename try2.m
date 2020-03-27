close all
clear all
%% Question1 partA
theta = 0:3:177;
iptsetpref('ImshowAxesVisible','on');
img=imread("../data/SheppLogan256.png");
%size(img)
[R,XP] = radon(img,theta);
subplot(5,2,1); imshow(img,[]);title("original image");
subplot(5,2,2); 
imshow(R,[],'Xdata',theta,'Ydata',XP);
xlabel('\theta (degrees)');
ylabel('t');
colormap(gca,hot), colorbar;
title("Radon Transform image");

back_image= mat2gray(iradon(R,theta,'none'));
subplot(5,2,3); imshow(back_image);title("Without Filter");

back_image= mat2gray(iradon(R,theta,'Ram-Lak',1,256));
subplot(5,2,5); imshow(back_image);title("Ram-Lak Filter with w=1*max");

back_image= mat2gray(iradon(R,theta,'Ram-Lak',0.5,256));
subplot(5,2,6); imshow(back_image);title("Ram-Lak Filter with w=0.5*max");

back_image= mat2gray(iradon(R,theta,'Shepp-Logan',1,256));
subplot(5,2,7); imshow(back_image);title("Shepp-Logan Filter with w=1*max");
back_image= mat2gray(iradon(R,theta,'Shepp-Logan',0.5,256));
subplot(5,2,8); imshow(back_image);title("Shepp-Logan Filter with w=0.5*max");

back_image= mat2gray(iradon(R,theta,'Cosine',1,256));
subplot(5,2,9); imshow(back_image);title("Cosine Filter with w=1*max");
back_image= mat2gray(iradon(R,theta,'Cosine',0.5,256));
subplot(5,2,10); imshow(back_image);title("Cosine Filter with w=0.5*max");


%% Question1 partB
S0=mat2gray(imread("../data/SheppLogan256.png"));
S1=mat2gray(imgaussfilt(S0,1));
S5=mat2gray(imgaussfilt(S0,5));
subplot(2,3,1);imshow(S0,[]);
subplot(2,3,2);imshow(S1,[]);
subplot(2,3,3);imshow(S5,[]);
[Rd0,XP1] = radon(S0,theta);
[Rd1,XP2] = radon(S1,theta);
[Rd5,XP3] = radon(S5,theta);

R0=mat2gray(iradon(Rd0,theta,'Ram-Lak',1,256));
R1= mat2gray(iradon(Rd1,theta,'Ram-Lak',1,256));
R5= mat2gray(iradon(Rd5,theta,'Ram-Lak',1,256));
subplot(2,3,4); imshow(R0,[]);title("S=0");
subplot(2,3,5); imshow(R1,[]);title("S=1");
subplot(2,3,6); imshow(R5,[]);title("S=5");
denom0=sqrt(sum(S0.^2,[1,2]))
denom1=sqrt(sum(S1.^2,[1,2]))
denom5=sqrt(sum(S5.^2,[1,2]))
RMSE0=sqrt(sum((S0-R0).^2,[1,2]))/denom0
RMSE1=sqrt(sum((S1-R1).^2,[1,2]))/denom1
RMSE5=sqrt(sum((S5-R5).^2,[1,2]))/denom5

%% Question1 partC Absolute 2
Freq_max=size(Rd0,1)
RMSE_curve_0=zeros(Freq_max,1);
RMSE_curve_1=zeros(Freq_max,1);
RMSE_curve_5=zeros(Freq_max,1);
j =1:Freq_max
n = 2^nextpow2(size(X,1))

for i =j
    R0=mat2gray(iradon(Rd0,theta,'Ram-Lak',i/Freq_max,256));
    R1=mat2gray(iradon(Rd1,theta,'Ram-Lak',i/Freq_max,256));
    R5=mat2gray(iradon(Rd5,theta,'Ram-Lak',i/Freq_max,256));
    RMSE_curve_0(i)=sqrt(sum((S0-R0).^2,[1,2]))/denom0;
    RMSE_curve_1(i)=sqrt(sum((S1-R1).^2,[1,2]))/denom1;
    RMSE_curve_5(i)=sqrt(sum((S5-R5).^2,[1,2]))/denom5;
end
figure;
title('RMSE For different frequencies');
subplot(1,3,1);plot(j, RMSE_curve_0);
subplot(1,3,2);plot(j, RMSE_curve_1);
subplot(1,3,3);plot(j, RMSE_curve_5);
%%
