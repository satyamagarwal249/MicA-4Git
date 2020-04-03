%% Q.2 (a)
% Build a CT imaging / system matrix, A
img = mat2gray(imread('../../data/ChestPhantom.png'));
I = zeros(128,128);
A = zeros(185*180,128*128,'single');
B = zeros(128,128);

for i=1:128
    for j=1:128
        B(i,j)=i+(j-1)*128;
    end
end

theta = 0:179;
for j = 1:128
    for i = 1:128
        I(i,j) = B(i,j);
        R = radon(I,theta);
        k = find(R>0);
        A(k, B(i,j)) = R(k)/B(i,j);
        I(i,j) = 0;
    end
end

%% Q.2 (b)
% Generate Radon Transform from matrix A & add Gaussian noise
col_img = reshape(img,[],1);
col_R = A*col_img;

minR = min(col_R);
maxR = max(col_R);
noise_std = (maxR - minR)*0.02;
noise_var = noise_std^2;

R = reshape(col_R,[],180);

figure;
imshow(R, []);
xlabel('\theta (degree)');
ylabel('t');
% colormap(gca, hot);
title('Radon Transform');
saveas(gcf, 'b1. Radon.jpg');
pause(1);

% noisyR = imnoise(R, 'gaussian', 0, noise_var);

noise_gauss = noise_std.*randn(size(R));
noisyR = R + noise_gauss;

% noisyR = imgaussfilt(R, noise_std);

figure;
imshow(noisyR, []);
xlabel('\theta (degree)');
ylabel('t');
% colormap(gca, hot);
title('Noisy Radon Transform');
saveas(gcf, 'b2. Noisy Radon.jpg');
pause(1);


%% Q.2 (c)
% Image Reconstruction from noisy data using filtered back projection
w_max = floor((size(noisyR,1) - 1)/2);
filtR = myFilter(noisyR, 'Ram-Lak', w_max);
filtBackProj = mat2gray(iradon(filtR, theta, 'none', 128));

figure;
imshow(filtBackProj, []);
title('Reconstructed Image from Noisy CT data');
saveas(gcf, 'c1. Reconstructed Image.jpg');
pause(1);

RRMSE = sqrt(sum(sum((img - filtBackProj).^2)))/sqrt(sum(sum(img.^2)));
disp(['RRMSE = ', num2str(RRMSE)]);


%% Q.2 (d)
% Tikhonov Regularized Reconstruction
%lambda = 0.01:0.01:0.99;
% lambda = 1;
% I = eye(size(A,2));
% col_noisyR = reshape(noisyR, [], 1);
% reconst_img = inv(A'*A + lambda*I) * A' * col_noisyR;
% reconst_img = reshape(reconst_img, [], 128);

% figure;
% imshow(reconst_img, []);
% title('Reconstructed Image by Tikhonov Regularization');
% pause(1);

% reconstRRMSE = sqrt(sum(sum((img - reconst_img).^2)))/sqrt(sum(sum(img.^2)));
% disp(['RRMSE = ', num2str(RRMSE)]);

disp('Q.2(d) started...');
I = eye(size(A,2));
col_noisyR = reshape(noisyR, [], 1);

lambda = 0.01:0.05:0.51;
minRRMSE = 10000;
optLambda = -1;
for l = lambda
    disp(l);
    P = A' * A + l*I;
    Q = A' * col_noisyR;
    % reconst_img = pinv(P)*Q;
    reconst_img = P\Q;
    reconst_img = reshape(reconst_img, [], 128);
    reconstRRMSE = sqrt(sum(sum((img - reconst_img).^2)))/sqrt(sum(sum(img.^2)));
    if reconstRRMSE < minRRMSE
        minRRMSE = reconstRRMSE;
        optLambda = l;
    end
end

disp(['minRRMSE = ', num2str(minRRMSE)]);
disp(['optLambda = ', num2str(optLambda)]);

%% Q.2 (e)
% Reconstruction using MRF Priors
% % alphaGamma = struct('g1',[0.5,0.5], 'g2',[0.5,0.5], 'g3',[0.5,0.5]);
% n = 30;


%% Filter  functions
function filtR = myFilter(R, filter, L)
    w_max = floor((size(R,1) - 1)/2);
    w_min = ceil((size(R,1) - 1)/2);
    w = [0:w_max -w_min:-1]';
    
    rectL = ones(length(w), 1);
    rectL(L+2:length(w)-L) = 0;

    switch filter
        case 'Ram-Lak'
            A_w = abs(w).*rectL;
        case 'Shepp-Logan'
            rad = 0.5*pi*w/L;
            A_w = (abs(w).*sin(rad).*rectL)./rad;
            A_w(1,1) = 0;
        case 'Cosine'
            rad = 0.5*pi*w/L;
            A_w = abs(w).*cos(rad).*rectL;
        otherwise
            disp('ERROR - Specify proper filter name..!!');
    end

    filtR = real(ifft(A_w.*fft(R)));
end