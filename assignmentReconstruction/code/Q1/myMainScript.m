%% Q.1 (a)
%% Radon Transform
theta = 0:3:177;
img = mat2gray(imread('../../data/SheppLogan256.png'));
[R, t] = radon(img, theta);		% # rows in R = 367
w_max = floor((size(R,1) - 1)/2);

figure;
imshow(img, []);
title('Original Image');
% saveas(gcf, 'a1. Original Image.jpg');
pause(1);

figure;
imshow(R, [], 'Xdata', theta, 'Ydata' , t, 'InitialMagnification', 'fit');
xlabel('\theta (degrees)');
ylabel('t');
colormap(gca,hot), colorbar;
title('Radon Transform');
% saveas(gcf, 'a2. Radon Transform.jpg');
pause(1);

%% Unfiltered Back Projection
unfiltBackProj = mat2gray(iradon(R, theta, 'none', 256));
figure;
imshow(unfiltBackProj, []);
title('Unfiltered Back Projection');
% saveas(gcf, 'a3. Unfiltered Back Projection.jpg');
pause(1);

%% Various Filtered Back Projections
filtR = myFilter(R, 'Ram-Lak', w_max);
filtBackProj = mat2gray(iradon(filtR, theta, 'none', 256));
figure;
imshow(filtBackProj, []);
title('Ram-Lak Filtered Back Projection (L=w_{max})');
% saveas(gcf, 'a4. Ram-Lak Filtered Back Projection (w_max).jpg');
pause(1);

filtR = myFilter(R, 'Ram-Lak', floor(w_max/2));
filtBackProj = mat2gray(iradon(filtR, theta, 'none', 256));
figure;
imshow(filtBackProj, []);
title('Ram-Lak Filtered Back Projection (L=w_{max}/2)');
% saveas(gcf, 'a5. Ram-Lak Filtered Back Projection (0.5w_max).jpg');
pause(1);

filtR = myFilter(R, 'Shepp-Logan', w_max);
filtBackProj = mat2gray(iradon(filtR, theta, 'none', 256));
figure;
imshow(filtBackProj, []);
title('Shepp-Logan Filtered Back Projection (L=w_{max})');
% saveas(gcf, 'a6. Shepp-Logan Filtered Back Projection (w_max).jpg');
pause(1);

filtR = myFilter(R, 'Shepp-Logan', floor(w_max/2));
filtBackProj = mat2gray(iradon(filtR, theta, 'none', 256));
figure;
imshow(filtBackProj, []);
title('Shepp-Logan Filtered Back Projection (L=w_{max}/2)');
% saveas(gcf, 'a7. Shepp-Logan Filtered Back Projection (0.5w_max).jpg');
pause(1);

filtR = myFilter(R, 'Cosine', w_max);
filtBackProj = mat2gray(iradon(filtR, theta, 'none', 256));
figure;
imshow(filtBackProj, []);
title('Cosine Filtered Back Projection (L=w_{max})');
% saveas(gcf, 'a8. Cosine Filtered Back Projection (w_max).jpg');
pause(1);

filtR = myFilter(R, 'Cosine', floor(w_max/2));
filtBackProj = mat2gray(iradon(filtR, theta, 'none', 256));
figure;
imshow(filtBackProj, []);
title('Cosine Filtered Back Projection (L=w_{max}/2)');
% saveas(gcf, 'a9. Cosine Filtered Back Projection (0.5w_max).jpg');
pause(1);

% The Unfiltered Back Projection is blurred out (as expected).

%% Q.1 (b)
%% Gaussian blurred images
S0 = img;
figure;
imshow(S0, []);
title('Shepp-Logan Image, S_0');
% saveas(gcf, 'b1. S0.jpg');
pause(1);

S1 = mat2gray(imgaussfilt(img,1));
figure;
imshow(S1, []);
title('Shepp-Logan Image, S_1');
% saveas(gcf, 'b2. S1.jpg');
pause(1);

S5 = mat2gray(imgaussfilt(img,5));
figure;
imshow(S5, []);
title('Shepp-Logan Image, S_5');
% saveas(gcf, 'b3. S5.jpg');
pause(1);

%% Radon Transform of Gaussian Blurred Images
[h0,t0] = radon(S0,theta);
[h1,t1] = radon(S1,theta);
[h5,t5] = radon(S5,theta);

%% Ram-Lak Filtered Back Projection of Radon Transform of Gaussian Blurred Images
filtR = myFilter(h0, 'Ram-Lak', w_max);
R0 = mat2gray(iradon(filtR, theta, 'none', 256));
denom = sqrt(sum(sum(S0.^2)));
RRMSE = sqrt(sum(sum((S0-R0).^2)))/denom;
figure;
imshow(R0, []);
title('Ram-Lak Filtered Back Projection, R_0');
% saveas(gcf, 'b4. R0.jpg');
pause(1);
disp(['RRMSE(S0, R0) = ', num2str(RRMSE)]);

filtR = myFilter(h1, 'Ram-Lak', w_max);
R1 = mat2gray(iradon(filtR, theta, 'none', 256));
denom = sqrt(sum(sum(S1.^2)));
RRMSE = sqrt(sum(sum((S1-R1).^2)))/denom;
figure;
imshow(R1, []);
title('Ram-Lak Filtered Back Projection, R_1');
% saveas(gcf, 'b5. R1.jpg');
pause(1);
disp(['RRMSE(S1, R1) = ', num2str(RRMSE)]);

filtR = myFilter(h5, 'Ram-Lak', w_max);
R5 = mat2gray(iradon(filtR, theta, 'none', 256));
denom = sqrt(sum(sum(S5.^2)));
RRMSE = sqrt(sum(sum((S5-R5).^2)))/denom;
figure;
imshow(R5, []);
title('Ram-Lak Filtered Back Projection, R_5');
% saveas(gcf, 'b6. R5.jpg');
pause(1);
disp(['RRMSE(S5, R5) = ', num2str(RRMSE)]);


%% Q.1 (c)
%% RRMSE v/s frequency plot
RRMSE0 = [];
for L = 1:w_max
    filtR = myFilter(h0, 'Ram-Lak', L);
    R0 = mat2gray(iradon(filtR, theta, 'none', 256));
    denom = sqrt(sum(sum(S0.^2)));
    e = sqrt(sum(sum((S0-R0).^2)))/denom;
    RRMSE0 = [RRMSE0 e];
end

RRMSE1 = [];
for L = 1:w_max
    filtR = myFilter(h1, 'Ram-Lak', L);
    R1 = mat2gray(iradon(filtR, theta, 'none', 256));
    denom = sqrt(sum(sum(S1.^2)));
    e = sqrt(sum(sum((S1-R1).^2)))/denom;
    RRMSE1 = [RRMSE1 e];
end

RRMSE5 = [];
for L = 1:w_max
    filtR = myFilter(h5, 'Ram-Lak', L);
    R5 = mat2gray(iradon(filtR, theta, 'none', 256));
    denom = sqrt(sum(sum(S5.^2)));
    e = sqrt(sum(sum((S5-R5).^2)))/denom;
    RRMSE5 = [RRMSE5 e];
end

figure;
plot(1:w_max, RRMSE0, 'r', 1:w_max, RRMSE1, 'g', 1:w_max, RRMSE5, 'b');
xlabel('Threshold Frequency, L');
ylabel('RRMSE');
legend('RRMSE(S_0, R_0)', 'RRMSE(S_1, R_1)', 'RRMSE(S_5, R_5)');
title('RRMSE v/s Threshold Frequency');
% saveas(gcf, 'c1. RRMSE_vs_L.jpg');
pause(1);


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