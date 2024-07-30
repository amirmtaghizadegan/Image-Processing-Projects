clear
close all
clc

image = imread("clown_noised.jpg");
image = double(image(:, :, 1));

f = fftshift(fft2(image));

mask = imread("mask4.png");
mask = double(mask(:, :, 1) > 0);
% imshow()

g = real(ifft2(ifftshift(mask .* f)));
figure, imshow(g, [])
figure, imshow(image, [])