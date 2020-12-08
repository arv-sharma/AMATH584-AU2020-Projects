%% YALE FACES

%% LOAD CROPPED DATA
clearvars; clc; close all

addpath('.\yalefaces_cropped\CroppedYale')
yale_cr_folders = dir('.\yalefaces_cropped\CroppedYale\yale*');

Imgs_cr = nan(192 * 168, length(yale_cr_folders) * 3);
Imgs_cr_montage = nan(192, 168, length(yale_cr_folders) * 3);

% Get one image from the 38 different subjects
for ii = 1:1:length(yale_cr_folders)
    filename = strcat(yale_cr_folders(ii).folder, '\', ...
        yale_cr_folders(ii).name, '\', yale_cr_folders(ii).name, ...
        '_P00A+000E+00.pgm');
    sub_dir_files = dir(strcat(yale_cr_folders(ii).folder, '\', ...
        yale_cr_folders(ii).name, '\', yale_cr_folders(ii).name, '*'));
    rand_file_nums = randperm(length(sub_dir_files) - 1, 2) + 1;    % to avoid ...
                                                                    % picking
                                                                    % same
                                                                    % files
    temp = imread(filename);
    Imgs_cr_montage(:, :, (ii - 1) * 3 + 1) = temp;
    Imgs_cr(:, (ii - 1) * 3 + 1) = temp(:);
    for jj = 1:1:2
        filename_rand = strcat(yale_cr_folders(ii).folder, '\', ...
            yale_cr_folders(ii).name, '\',...
            sub_dir_files(rand_file_nums(jj)).name);
        temp = imread(filename_rand);
        Imgs_cr_montage(:, :, (ii - 1) * 3 + jj + 1) = temp;
        Imgs_cr(:, (ii - 1) * 3 + jj + 1) = temp(:);
    end
end

figure, montage(uint8(Imgs_cr_montage(:, :, randperm(size(Imgs_cr, 2), 36))),...
    'Size', [6, 6])

% Using 90% of images for training
num_test_imgs = round(0.9 * size(Imgs_cr, 2));
mean_face_cr = mean(Imgs_cr(:, 1:num_test_imgs), 2);
X_Imgs_cr = Imgs_cr(:, 1:num_test_imgs) - mean_face_cr;

%% SVD
[U_cr, S_cr, V_cr] = svd(X_Imgs_cr, 'econ');
sig_cr = diag(S_cr);

U_cr_rs = nan(size(U_cr));
for jj = 1:1:num_test_imgs
    U_cr_rs(:, jj) = rescale(U_cr(:, jj), 0, 255);
end
U_cr_rs_mon = reshape(U_cr_rs, [192, 168, num_test_imgs]);
U_cr_rs_mon = cat(3, U_cr_rs_mon);
figure, montage(uint8(U_cr_rs_mon(:, :, 1:36)), 'Size', [6, 6])

Y_cr = V_cr * X_Imgs_cr';

fig1 = figure;
fig1.Units = 'inches';
fig1.Position = [-.1 1.8 6.75 5.0625];
fig1.PaperUnits = 'inches';
fig1.PaperSize = [6.75 5.0625];

s1 = subplot(2, 1, 1);
s1.Box = 'on';
hold on
h1 = plot(sig_cr, 'ko-', 'LineWidth', 1.1);
h1.MarkerSize = 3.5;
xlabel({'Mode, k'; '(a)'})
ylabel('Singular value, \sigma_k')
axis tight
s1.YScale = 'log';

s2 = subplot(2, 1, 2);
s2.Box = 'on';
hold on
h2 = plot(cumsum(sig_cr) / sum(sig_cr), 'ks-', 'LineWidth', 1.1);
h2.MarkerSize = 3.5;
xlabel({'Mode, k'; '(b)'})
ylabel({'Cumulative energy in', 'first k modes, \sigma_k'})
axis tight

%% Power iterating to find dominant eigenvalue and eigenvector

A = X_Imgs_cr * X_Imgs_cr';
lambda_est = [];
Tol = 1e-8;
iters = 0;
error = 1;

while error > Tol
    iters = iters + 100;
    [lambda_est, u_est] = power_iter(A, iters, 1);
    error = min([abs(lambda_est(end) - S_cr(1) ^ 2), norm(U_cr(:, 1) - u_est),...
        norm(-U_cr(:, 1) - u_est)]);
end

fprintf('\nEigenvalue from power iteration = %.4g\n', lambda_est)
fprintf('\nEigenvalue from square of dominant singular value = %.4g\n',...
    S_cr(1) ^ 2)
fprintf('\nError in eigenvalue between the methods = %.4g\n',...
    abs(lambda_est - S_cr(1) ^ 2))
fprintf('\nnorm(Dominant eigenvector - Leading vector from U matrix) = %.4g\n',...
    min([norm(U_cr(:, 1) - u_est), norm(-U_cr(:, 1) - u_est)]))

fig = figure;
fig.Units = 'inches';
fig.Position = [-.1 1.8 6.5 3];
fig.PaperUnits = 'inches';
fig.PaperSize = [6.5 3];
s1 = subplot(1, 2, 1);
imagesc(reshape(-U_cr(:, 1), [192, 168]))
colormap(gray)
s1.XTick = [];
s1.YTick = [];
xlabel({'(a)'; 'From SVD'})
s2 = subplot(1, 2, 2);
imagesc(reshape(u_est, [192, 168]))
s2.XTick = [];
s2.YTick = [];
xlabel({'(b)'; 'From power iteration'})
% savefig(fig, 'yale_dom_mode_comp')
% print('yale_dom_mode_comp', '-depsc', '-r300')

%% Randomized Sampling

[m, n] = size(X_Imgs_cr);
K = 10;
Omega = randn(n, K);
Y = X_Imgs_cr * Omega;

[Q, R] = qr(Y, 0);

B = Q' * X_Imgs_cr;
[u, s, v] = svd(B, 'econ');

u_approx = Q * u;

fig2 = figure;
fig2.Units = 'inches';
fig2.Position = [-.1 1.8 6.5 8];
fig2.PaperUnits = 'inches';
fig2.PaperSize = [6.5 8];
s1 = subplot(3, 2, 1);
imagesc(reshape(-U_cr(:, 1), [192, 168]))
colormap(gray)
s1.XTick = [];
s1.YTick = [];
xlabel({'(a)'; 'Mode 1 from SVD'})
s2 = subplot(3, 2, 2);
imagesc(reshape(-u_approx(:, 1), [192, 168]))
s2.XTick = [];
s2.YTick = [];
xlabel({'(b)'; 'Mode 1 from random sampling'})
s3 = subplot(3, 2, 3);
imagesc(reshape(-U_cr(:, 2), [192, 168]))
colormap(gray)
s1.XTick = [];
s1.YTick = [];
xlabel({'(c)'; 'Mode 2 from SVD'})
s4 = subplot(3, 2, 4);
imagesc(reshape(-u_approx(:, 2), [192, 168]))
s4.XTick = [];
s4.YTick = [];
xlabel({'(d)'; 'Mode 2 from random sampling'})
s5 = subplot(3, 2, 5);
imagesc(reshape(-U_cr(:, 3), [192, 168]))
colormap(gray)
s1.XTick = [];
s1.YTick = [];
xlabel({'(e)'; 'Mode 3 from SVD'})
s6 = subplot(3, 2, 6);
imagesc(reshape(-u_approx(:, 3), [192, 168]))
s6.XTick = [];
s6.YTick = [];
xlabel({'(f)'; 'Mode 3 from random sampling'})
% savefig(fig2, 'Random_sampling_modes')
% print('Random_sampling_modes', '-depsc', '-r300')

colorvec = hsv(4);
c1 = 1;
for K = [2, 5, 10, 20]
    Omega = randn(n, K);
    Y = X_Imgs_cr * Omega;

    [Q, R] = qr(Y, 0);

    B = Q' * X_Imgs_cr;
    [u, s, v] = svd(B, 'econ');
    sig_cr_approx = diag(s);
    
    subplot(s1);
    plot(s, 'Marker', 'o', 'Color', colorvec(c1, :), 'LineStyle', 'none',...
        'MarkerSize', 8)
    xlim([1, 30])
    
    subplot(s2);
    plot(cumsum(s) / sum(s), 'Marker', 's', 'Color', colorvec(c1, :),...
        'LineStyle', '-', 'MarkerSize', 8)
    xlim([1, 30])
    
    c1 = c1 + 1;
end

subplot(s1);
legend({'Full SVD', 'K = 2', 'K = 5', 'K = 10', 'K = 20'})

subplot(s2);
legend({'Full SVD', 'K = 2', 'K = 5', 'K = 10', 'K = 20'})

% savefig(fig1, 'sing_val_decay')
% print('sing_val_decay', '-depsc', '-r300')

