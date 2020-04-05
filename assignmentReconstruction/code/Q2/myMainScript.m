%% Q.2 (a)
% Build a CT imaging / system matrix, A
img = mat2gray(imread('../../data/ChestPhantom.png'));
global w;
global h;
h = size(img,1);
w = size(img,2);
theta = 0:179;
row = size(radon(img, theta), 1);

I = zeros(h,w);
A = zeros(row*length(theta), h*w, 'single');
B = zeros(h,w);

for i=1:h
    for j=1:w
        B(i,j)=i+(j-1)*128;
    end
end

for j = 1:w
    for i = 1:h
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

R = reshape(col_R, [], length(theta));

figure;
imshow(R, []);
xlabel('\theta (degree)');
ylabel('t');
colormap(gca, hot);
title('Radon Transform');
saveas(gcf, 'b1. Radon.jpg');
pause(1);

rng(77);
noise_gauss = noise_std.*randn(size(R));
noisyR = R + noise_gauss;

figure;
imshow(noisyR, []);
xlabel('\theta (degree)');
ylabel('t');
colormap(gca, hot);
title('Noisy Radon Transform');
saveas(gcf, 'b2. Noisy Radon.jpg');
pause(1);

%% Q.2 (c)
% Image Reconstruction from noisy data using filtered back projection
w_max = floor((size(noisyR,1) - 1)/2);
filtR = myFilter(noisyR, 'Ram-Lak', w_max);
filtBackProj = mat2gray(iradon(filtR, theta, 'none', h));

figure;
imshow(filtBackProj, []);
title('Filtered Back Projection from Noisy CT data');
saveas(gcf, 'c1. FilteredBackProjection.jpg');
pause(1);

RRMSE = sqrt(sum(sum((img - filtBackProj).^2)))/sqrt(sum(sum(img.^2)));
fprintf('Q.2(c)\nRRMSE = %f\n', RRMSE);


%% Q.2 (e)
% Reconstruction using MRF Priors
fprintf('\nQ2.(e)');
n = 30;
bckgrndNoisy = reshape(filtBackProj(1:n,1:n), [], 1);
sigma = std(bckgrndNoisy, 1);

filtBackProj = reshape(filtBackProj, [], 1);
img = reshape(img, [], 1);
fprintf('\nQuadratic Prior Reconstruction\n');
reconst_img1 = reshape(denoise(filtBackProj, img, sigma, 'Squared'), [], w);
fprintf('\nHuber Prior Reconstruction\n');
reconst_img2 = reshape(denoise(filtBackProj, img, sigma, 'Huber'), [], w);
fprintf('\nDiscontinuity-Adaptive Prior Reconstruction\n');
reconst_img3 = reshape(denoise(filtBackProj, img, sigma, 'DiscAdapt'), [], w);

figure();
imshow(reconst_img1, []);
title('Quadratic Prior Reconstruction');
saveas(gcf, 'e1. Quadratic Prior Reconstruction.jpg');
pause(1);

figure();
imshow(reconst_img2, []);
title('Huber Prior Reconstruction');
saveas(gcf, 'e2. Huber Prior Reconstruction.jpg');
pause(1);

figure();
imshow(reconst_img3, []);
title('Discontinuity-Adaptive Prior Reconstruction');
saveas(gcf, 'e3. Discontinuity-Adaptive Prior Reconstruction.jpg');

% %% Q.2 (d)
% % Tikhonov Regularized Reconstruction
I = eye(size(A,2));
col_noisyR = reshape(noisyR, [], 1);

lambda = 0.01:0.05:0.01;
minRRMSE = 10000;
optLambda = -1;

% Closed form Solution
% for l = lambda
%     P = A' * A + (l^2)*I;
%     Q = A' * col_noisyR;
%     % disp([l, min(min(P)), max(max(P)), det(P)]);
%     % reconst_img = pinv(P)*Q;
%     reconst_img = P\Q;
%     reconst_img = reshape(reconst_img, [], 128);
%     reconstRRMSE = sqrt(sum(sum((img - reconst_img).^2)))/sqrt(sum(sum(img.^2)));
%     disp(['Reconstructed RRMSE = ', num2str(reconstRRMSE)]);
%     if reconstRRMSE < minRRMSE
%         minRRMSE = reconstRRMSE;
%         optLambda = l;
%     end
% end

% Iterative Solution
for l = lambda
    x = reshape(rand(128),[],1);
    prevGradMag = inf;
    lr = 0.01;
    epoch = 1;
    while prevGradMag > 0.01
        currGrad = 2*((A'*A + (l^2)*I)*x - A'*col_noisyR);
        x = x - lr*currGrad;
        prevGradMag = norm(currGrad);
        disp([epoch, prevGradMag]);
        epoch = epoch + 1;
    end

    reconst_img = reshape(x, [], 128);
    reconstRRMSE = sqrt(sum(sum((img - reconst_img).^2)))/sqrt(sum(sum(img.^2)));
    disp(['Reconstructed RRMSE = ', num2str(reconstRRMSE)]);
    if reconstRRMSE < minRRMSE
        minRRMSE = reconstRRMSE;
        optLambda = l;
    end
end

% disp(['minRRMSE = ', num2str(minRRMSE)]);
% disp(['optLambda = ', num2str(optLambda)]);


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

% Denoise function
function X = denoise(noisyImg, pureImg, sigma, MRF)
    switch MRF
        case 'Squared'
            a = [0.79, 1.0, 0.99];  % Optimum values of alpha
            for i = 1:length(a)
                stepSize = 0.1;
                X = noisyImg;
                prevLoss = calcLoss(X, noisyImg, sigma);
                while stepSize > 1e-10
                    grPrior = gradMRF_g1(X);
                    grLikelihood = 2*(X - noisyImg)/(sigma^2);
                    grad = a(i)*grPrior + (1-a(i))*grLikelihood;

                    X = X - stepSize*grad;
                    currLoss = calcLoss(X, noisyImg, sigma);
                    
                    if currLoss < prevLoss
                        stepSize = stepSize * 1.1;
                    else
                        stepSize = stepSize * 0.5;
                    end

                    prevLoss = currLoss;
                    RRMSE = sqrt(sum((pureImg - X).^2))/sqrt(sum(pureImg.^2));
                end
                fprintf('RRMSE (at alpha=%f) = %f\n', a(i), RRMSE);
            end
    
        case 'Huber'
            a = [0.08, 0.12, 0.1];
            g = [0.08, 0.12, 0.1];
            
            for i = 1:length(a)
                stepSize = 0.1;
                X = noisyImg;
                prevLoss = calcLoss(X, noisyImg, sigma);
                while stepSize > 1e-8
                    grPrior = gradMRF_g2(X, g(i));
                    grLikelihood = 2*(X - noisyImg)/(sigma^2);
                    grad = a(i)*grPrior + (1-a(i))*grLikelihood;

                    X = X - stepSize*grad;
                    currLoss = calcLoss(X, noisyImg, sigma);
                    
                    if currLoss < prevLoss
                        stepSize = stepSize * 1.1;
                    else
                        stepSize = stepSize * 0.5;
                    end
                    
                    prevLoss = currLoss;
                    RRMSE = sqrt(sum((pureImg - X).^2))/sqrt(sum(pureImg.^2));
                end
                fprintf('RRMSE (at alpha=%f, gamma=%f) = %f\n', a(i), g(i), RRMSE);
            end

        case 'DiscAdapt'
            a = [0.79, 1.0, 0.99];
            g = [0.79, 1.0, 0.99];

            for i = 1:length(a)
                stepSize = 0.1;
                X = noisyImg;
                prevLoss = calcLoss(X, noisyImg, sigma);
                while stepSize > 1e-8
                    % disp(stepSize);
                    grPrior = gradMRF_g3(X, g(i));
                    grLikelihood = 2*(X - noisyImg)/(sigma^2);
                    grad = a(i)*grPrior + (1-a(i))*grLikelihood;

                    X = X - stepSize*grad;
                    currLoss = calcLoss(X, noisyImg, sigma);
                    
                    if currLoss < prevLoss
                        stepSize = stepSize * 1.1;
                    else
                        stepSize = stepSize * 0.5;
                    end
                    
                    prevLoss = currLoss;
                    RRMSE = sqrt(sum((pureImg - X).^2))/sqrt(sum(pureImg.^2));
                end
                fprintf('RRMSE (at alpha=%f, gamma=%f) = %f\n', a(i), g(i), RRMSE);
            end
    end
end

function grPrior = gradMRF_g1(X)
    global w;
    global h;
    X = reshape(X, [], w);
    N = zeros(h, w);
    N = N + 4*X;
    N = N - circshift(X, [1,0]);
    N = N - circshift(X, [0,1]);
    N = N - circshift(X, [-1,0]);
    N = N - circshift(X, [0,-1]);
    N = 2*N;
    grPrior = reshape(N, [], 1);
end

function grPrior = gradMRF_g2(X, g)
    global w;
    global h;
    X = reshape(X, [], w);
    N = zeros(h, w);

    T = circshift(X, [1,0]) - X;
    b = abs(T) > g;
    s = T(b);
    T(b) = g*(s./abs(s));
    N = N + T;

    T = circshift(X, [0,1]) - X;
    b = abs(T) > g;
    s = T(b);
    T(b) = g*(s./abs(s));
    N = N + T;

    T = circshift(X, [-1,0]) - X;
    b = abs(T) > g;
    s = T(b);
    T(b) = g*(s./abs(s));
    N = N + T;

    T = circshift(X, [0,-1]) - X;
    b = abs(T) > g;
    s = T(b);
    T(b) = g*(s./abs(s));
    N = N + T;

    grPrior = reshape(N, [], 1);
end

function grPrior = gradMRF_g3(X, g)
    global w;
    global h;
    X = reshape(X, [], w);
    N = zeros(h, w);
    T = X - circshift(X, [1,0]);
    N = N + ((g*T)./(g + abs(T)));
    T = X - circshift(X, [0,1]);
    N = N + ((g*T)./(g + abs(T)));
    T = X - circshift(X, [-1,0]);
    N = N + ((g*T)./(g + abs(T)));
    T = X - circshift(X, [0,-1]);
    N = N + ((g*T)./(g + abs(T)));
    grPrior = reshape(N, [], 1);
end

function totLoss = calcLoss(X, noisyImg, sigma)
    global w;
    X = reshape(X, [], w);
    noisyImg = reshape(noisyImg, [], w);
    totCliquePot = 0;

    N = X - circshift(X, [1,0]);
    totCliquePot = totCliquePot + sum(sum(N*N));

    N = X - circshift(X, [0,1]);
    totCliquePot = totCliquePot + sum(sum(N*N));

    lossLikelihood = sum(sum(((X - noisyImg)/sigma).^2));
    totLoss = totCliquePot + lossLikelihood;
end