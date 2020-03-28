img=mat2gray(imread("../data/ChestPhantom.png"));
imshow(img);
img_size=size(img,1);%128
Ro = radon(img,0:179);
x=mat2gray(iradon(Ro,0:179));
imshow(x);
R_size=size(Ro,1) % value is 185
I=zeros(128,128);
A=zeros(185*180,128*128,'single');
B=zeros(128,128);
for i=1:128
    for j=1:128
        B(i,j)=i+(j-1)*128;
    end
end
% find(Ro>0)
% A=[1 2; 1 2 ; 2 2 ;4 5]
% A(find(A>1))

theta=0:179
for i=1:128
    for j=1:128
        temp=B(i,j);
        I(i,j)=temp;
        R=radon(I,theta);
            k =find(R>0);
%             A(k, temp)=R(fix(k/185)+1,mod(k,185)+1)/temp;
            A(k, temp)=R(k)/temp;
        I(i,j)=0;
    end
    i
end



%C=A(1:185*180,:);
COl_img=reshape(img,[],1);
R1=A*COl_img;

% imshow(R1);

mini=min(R1,[],'all')
maxa=max(R1,[],'all')
noise_sigma=(maxa-mini)*0.02;
noisyR1=imgaussfilt(R1,noise_sigma);
noisyR1=reshape(noisyR1,[],180);

back_image= mat2gray(iradon(noisyR1,0:179,'cosine',128));
imshow(back_image);
img=mat2gray(img);

RMSE=sqrt(sum((img-back_image).^2,[1,2]))/sqrt(sum(img.^2,[1,2]));
    
%m=A'*A;



% 
% C=A(1:185*180,:);
% COl_img=reshape(img,[],1)
% R1=C*COl_img;
% R1=reshape(R1,[],180);
% imshow(R1);
% x1=mat2gray(iradon(R1,0:180));
% imshow(x1);