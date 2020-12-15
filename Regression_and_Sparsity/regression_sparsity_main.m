clearvars
close all

%% Read training and test sets with labels

Train_data = read_MNIST('./train-images.idx3-ubyte');
Train_labels = read_MNIST('./train-labels.idx1-ubyte');
Test_data = read_MNIST('./t10k-images.idx3-ubyte');
Test_labels = read_MNIST('./t10k-labels.idx1-ubyte');
num_train_images = length(Train_labels);
num_test_images = length(Test_labels);

%% Pre-processing the images to make them binary images

Train_data_bin = nan(size(Train_data));
Test_data_bin = nan(size(Test_data));
for ii = 1: 1: num_train_images
    temp = reshape(Train_data(:, ii), [28, 28]);
    temp = imbinarize(uint8(temp)); % Threshold given by Otsu's method
    Train_data_bin(:, ii) = temp(:);
    if ii <= num_test_images
        temp = reshape(Test_data(:, ii), [28, 28]);
        temp = imbinarize(uint8(temp)); % Threshold given by Otsu's method
        Test_data_bin(:, ii) = temp(:);
    end        
end

Train_data_bin = transpose(Train_data_bin);
Test_data_bin = transpose(Test_data_bin);

% Visualizing training and testing data
I_mon = reshape([Train_data(:, 1:18), Test_data(:, 1:18)],...
    [28, 28, 36]);
I_mon = cat(3, uint8(I_mon));
I_mon_bin = reshape(transpose([Train_data_bin(1:18, :); Test_data_bin(1:18, :)]),...
    [28, 28, 36]);
I_mon_bin = cat(3, uint8(I_mon_bin * 255));

fig1 = figure;
fig1.Units = 'inches';
fig1.Position = [-.1 1.8 6 4.5];
fig1.PaperUnits = 'inches';
fig1.PaperSize = [6 4.5];
s1 = subplot(1, 2, 1);
montage(I_mon)
xlabel('(a)')
s2 = subplot(1, 2, 2);
montage(I_mon_bin)
xlabel('(b)')
% savefig(fig1, 'raw_bin_images')
% print('raw_bin_images', '-depsc', '-r300')

%% Modifying labels

lookup_table = eye(10);
Train_labels_mod = nan(num_train_images, 10);
Test_labels_mod = nan(num_test_images, 10);

for ii = 1: 1: num_train_images
    if Train_labels(ii) == 0
        Train_labels_mod(ii, :) = lookup_table(10, :);
    else
        Train_labels_mod(ii, :) = lookup_table(Train_labels(ii), :);
    end
    if ii <= num_test_images
        if Test_labels(ii) == 0
            Test_labels_mod(ii, :) = lookup_table(10, :);
        else
            Test_labels_mod(ii, :) = lookup_table(Test_labels(ii), :);
        end
    end
end

%% Comparison of different methods to solve Ax = B

linSols = compAxBSolvers(Train_data_bin, Train_labels_mod,...
    Test_data_bin, Test_labels_mod, 0.02, 0.7);

% Accuracy and error
fig2 = figure;
fig2.Units = 'inches';
fig2.Position = [-.1 1.8 6 4.5];
fig2.PaperUnits = 'inches';
fig2.PaperSize = [6 4.5];
s1 = subplot(1, 2, 1);
s1.Box = 'on';
h1 = bar(linSols.errors, 'FaceColor', 'k');
ylim([0 1])
xlabel('Solver type')
ylabel('Normalized error measure')
s2 = subplot(1, 2, 2);
s2.Box = 'on';
h2 = bar(linSols.accuracy, 'FaceColor', 'k');
ylim([0 1])
xlabel('Solver type')
ylabel('Normalized accuracy')
% savefig(fig2, 'err_acc_overall')
% print('err_acc_overall', '-depsc', '-r300')

% Visualizing regression coefficients from different methods
labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)',...
    '(i)', '(j)'};
fig3 = figure;
fig3.Units = 'inches';
fig3.Position = [-.1 1.8 6 4.5];
fig3.PaperUnits = 'inches';
fig3.PaperSize = [6 4.5];
for ii = 1: 1: 6
    s = subplot(2, 3, ii);
    temp = linSols.(['X', num2str(ii)]);
    imagesc(reshape(temp(:, 3), [28, 28]))
    xlabel(labels{ii})
    axis square
    colormap(spring)
    s.XTick = [];
    s.YTick = [];
end
% savefig(fig3, 'compare_types')
% print('compare_types', '-depsc', '-r300')

% Comparison of mode shapes of different digits from Lasso with L2

fig4 = figure;
fig4.Units = 'inches';
fig4.Position = [-.1 1.8 6 4.5];
fig4.PaperUnits = 'inches';
fig4.PaperSize = [6 4.5];
for ii = 1: 1: 10
    s = subplot(3, 4, ii);
    temp = linSols.X4;
    imagesc(reshape(temp(:, ii), [28, 28]))
    xlabel(labels{ii})
    axis square
    colormap(spring)
    s.XTick = [];
    s.YTick = [];
end
% savefig(fig4, 'compare_num_modes_LassoL2')
% print('compare_num_modes_LassoL2', '-depsc', '-r300')

%% Finding important pixels for each digit

impPixels = struct;
impPixels.allDigits = [];
for kk = 1: 1: 10
    if kk ~= 10
        impPixels.(['dig', num2str(kk)]) = find(linSols.X4(:, kk));
        impPixels.allDigits = union(impPixels.allDigits,...
            impPixels.(['dig', num2str(kk)]));
    else
        impPixels.dig0 = find(linSols.X4(:, kk));
        impPixels.allDigits = union(impPixels.allDigits, impPixels.dig0);
    end
end

%% Finding error and accuracy with the reduced pixel model

errorsRedModel = nan(6, 1);
accuracyRedModel = nan(6, 1);

for ii = 1: 1: 6
    tempX = linSols.(['X', num2str(ii)]);
    temp_mod = Test_data_bin(:, impPixels.allDigits) * tempX(impPixels.allDigits, :);
    if ii == 3
        temp_mod = temp_mod + linSols.lasso1_int;
    elseif ii == 4
        temp_mod = temp_mod + linSols.lasso2_int;
    elseif ii == 5
        temp_mod = temp_mod + linSols.Robust_const;
    elseif ii == 6
        temp_mod = temp_mod + linSols.Ridge_const;
    end
    temp = rowMax(temp_mod);
    errorsRedModel(ii, 1) = norm(Test_labels_mod - temp) / norm(Test_labels_mod);
    accuracyRedModel(ii, 1) = nnz(all(Test_labels_mod == temp, 2)) /...
        size(Test_labels_mod, 1);
end
    
% Comparison with non-reduced model
fig5 = figure;
fig5.Units = 'inches';
fig5.Position = [-.1 1.8 6 4.5];
fig5.PaperUnits = 'inches';
fig5.PaperSize = [6 4.5];
s1 = subplot(1, 2, 1);
s1.Box = 'on';
h1 = bar([linSols.errors, errorsRedModel]);
xlabel('Solver type')
ylabel('Normalized error measure')
legend('Pixels = 784', ['Pixels = ', num2str(length(impPixels.allDigits))])
s2 = subplot(1, 2, 2);
s2.Box = 'on';
h2 = bar([linSols.accuracy, accuracyRedModel]);
ylim([0 1])
xlabel('Solver type')
ylabel('Normalized accuracy')
legend('Pixels = 784', ['Pixels = ', num2str(length(impPixels.allDigits))])
% savefig(fig5, 'err_acc_red')
% print('err_acc_red', '-depsc', '-r300')

%% Performing analysis on each individual digit

errorsDigits = struct('mod1', [], 'mod2', [], 'mod3', [], 'mod4', [],...
    'mod5', [], 'mod6', []);
accuracyDigits = struct('mod1', [], 'mod2', [], 'mod3', [], 'mod4', [],...
    'mod5', [], 'mod6', []);
for jj = 1: 1: 10
    if jj ~= 10
        output = compAxBSolversDigits(Train_data_bin,...
            double(Train_labels == jj), Test_data_bin,...
            double(Test_labels == jj), 0.02, 0.7);
    else
        output = compAxBSolversDigits(Train_data_bin,...
            double(Train_labels == 0), Test_data_bin,...
            double(Test_labels == 0), 0.02, 0.7);
    end
    
    % Comparing with the reduced model
    errorsRedModel = nan(6, 1);
    accuracyRedModel = nan(6, 1);

    for ii = 1: 1: 6
        tempX = output.(['X', num2str(ii)]);
        temp_mod = Test_data_bin(:, impPixels.allDigits) * tempX(impPixels.allDigits);
        if ii == 3
            temp_mod = temp_mod + output.lasso1_int;
        elseif ii == 4
            temp_mod = temp_mod + output.lasso2_int;
        elseif ii == 5
            temp_mod = temp_mod + output.Robust_const;
        elseif ii == 6
            temp_mod = temp_mod + output.Ridge_const;
        end
        temp = temp_mod;
        temp(temp < 0.5) = 0;
        temp(temp >= 0.5) = 1;
        errorsRedModel(ii, 1) = norm(double(Test_labels == 0) - temp) /...
            norm(double(Test_labels == 0));
        accuracyRedModel(ii, 1) = nnz(double(Test_labels == 0) == temp) /...
            size(Test_labels, 1);
        temp1 = errorsDigits.(['mod', num2str(ii)]);
        errorsDigits.(['mod', num2str(ii)]) = [temp1; [output.errors(ii),...
            errorsRedModel(ii)]];
        temp2 = accuracyDigits.(['mod', num2str(ii)]);
        accuracyDigits.(['mod', num2str(ii)]) = [temp2; [output.accuracy(ii),...
            accuracyRedModel(ii)]];
    end
    
end

% Comparison of results with different models for all 10 digits

fig6 = figure;
fig6.Units = 'inches';
fig6.Position = [-.1 1.8 6 4.5];
fig6.PaperUnits = 'inches';
fig6.PaperSize = [6 4.5];
fig7 = figure;
fig7.Units = 'inches';
fig7.Position = [-.1 1.8 6 4.5];
fig7.PaperUnits = 'inches';
fig7.PaperSize = [6 4.5];
for ii = 1: 1: 6
    figure(fig6);
    s = subplot(2, 3, ii);
    temp = errorsDigits.(['mod', num2str(ii)]);
    h = bar(temp);
    xlabel({'Digit'; labels{ii}})
    ylabel('Normalized error measure')
    
    figure(fig7);
    s = subplot(2, 3, ii);
    temp = accuracyDigits.(['mod', num2str(ii)]);
    h = bar(temp);
    xlabel({'Digit'; labels{ii}})
    ylabel('Normalized accuracy')
    
end
% savefig(fig6, 'err_digits')
% print('err_digits', '-depsc', '-r300')
% savefig(fig7, 'acc_digits')
% print('acc_digits', '-depsc', '-r300')
