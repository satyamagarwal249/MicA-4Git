%% Q.3 (a)
%The minimum RMSE giving theta sets for image-1(ChestCT) is 9 and for
%image-2(SheppLogan256) is 19.

% Plot of RRMSE v/s theta
img1 = mat2gray(imread('../../data/ChestCT.png'));
img_size1 = size(img1,1);
range = 0:150;
denominator = sqrt(sum(sum(img1.^2)));
RRMSE1 = [];

for i = 0:180
    theta = range+i;
    R = radon(img1,theta);
    reconstructed_img = mat2gray(iradon(R,theta,'Ram-Lak',1,img_size1));
    e = sqrt(sum(sum((img1-reconstructed_img).^2)))/denominator;
    RRMSE1 = [RRMSE1 e];
end

img2 = mat2gray(imread('../../data/SheppLogan256.png'));
img_size2 = size(img2,1);
range = 0:150;
denominator = sqrt(sum(sum(img2.^2)));
RRMSE2 = [];

for i = 0:180
    theta = range+i;
    R = radon(img2,theta);
    reconstructed_img = mat2gray(iradon(R,theta,'Ram-Lak',1,img_size2));
    e = sqrt(sum(sum((img2-reconstructed_img).^2)))/denominator;
    RRMSE2 = [RRMSE2 e];
end

figure();
plot(0:180, RRMSE1, 'r', 0:180, RRMSE2, 'b');
xlabel('\theta (degree)');
ylabel('RRMSE');
legend('ChestCT.png', 'SheppLogan256.png');
title('RRMSE v/s \theta (degree)');
%saveas(gcf, 'a1. RRMSEvsTheta.jpg');
pause(1);

[minRRMSE1 id1] = min(RRMSE1);
opt_theta1 = id1 - 1;
[minRRMSE2 id2] = min(RRMSE2);
opt_theta2 = id2 - 1;
disp([opt_theta1, opt_theta2]);

%% Q.3 (b)
% Reconstructed Images with least RRMSE
theta = range+opt_theta1;
R = radon(img1,theta);
reconstructed_img1 = mat2gray(iradon(R,theta,'Ram-Lak',1,img_size1));

theta = range+opt_theta2;
R = radon(img2,theta);
reconstructed_img2 = mat2gray(iradon(R,theta,'Ram-Lak',1,img_size2));

figure();
imshow(reconstructed_img1, []);
title('Reconstructed ChestCT with least RRMSE');
%saveas(gcf, 'b1. Reconstructed ChestCT.jpg');
pause(1);

figure();
imshow(reconstructed_img2, []);
title('Reconstructed SheppLogan256 with least RRMSE');
%saveas(gcf, 'b2. Reconstructed SheppLogan256.jpg');