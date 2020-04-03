%% Q.3 (a)
% Plot of RRMSE v/s theta
img = mat2gray(imread("../../data/ChestCT.png"));
img_size = size(img,1);
range = 0:150;
denominator = sqrt(sum(sum(img.^2)));
RRMSE = [];
opt_theta = 0;

for i = 0:180
    theta = range+i;
    R = radon(img,theta);
    reconstructed_img = mat2gray(iradon(R,theta,'Ram-Lak',1,img_size));
    RMSE_curve(i+1) = sqrt(sum(sum((img-reconstructed_img).^2)))/denominator;
    if RMSE_curve(max_theta+1) < RMSE_curve(i+1)
        max_theta = i;
    end
end

figure(1);
title('RRMSE For different thetas');
plot(0:180, RMSE_curve);
fprintf("max theta giving is %d",max_theta);

%% Q4 partB
figure(2);
theta=range+max_theta;
[R,XP] = radon(img,theta);
reconstructed_img=mat2gray(iradon(R,theta,'Ram-Lak',1,img_size));
imshow(reconstructed_img)
%% IN Image 2 same thing
img=mat2gray(imread("../data/SheppLogan256.png"));
img_size=size(img,1);
range=0:149;
denominator=sqrt(sum(img.^2,[1,2]));
RMSE_curve=zeros(180,1);
max_theta=0
for i= 0:179
    theta=range+i;
    R = radon(img,theta);
    reconstructed_img=mat2gray(iradon(R,theta,'Ram-Lak',1,img_size));
    RMSE_curve(i+1)=sqrt(sum((img-reconstructed_img).^2,[1,2]))/denominator;
    if RMSE_curve(max_theta+1)<RMSE_curve(i+1)
        max_theta=i;
    end
end
figure(3);
title('RMSE For different thetas');
plot(0:179, RMSE_curve);
fprintf("max theta giving is %d",max_theta);
%% Q4 partB
figure(4);
theta=range+max_theta;
[R,XP] = radon(img,theta);
reconstructed_img=mat2gray(iradon(R,theta,'Ram-Lak',1,img_size));
imshow(reconstructed_img)
