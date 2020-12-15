function results = compAxBSolvers(Train_dat, Train_lab,...
    Test_dat, Test_lab, lam1, lam2)

    % Comparison of 6 different methods to solve Ax = B
    
    results = struct;
    results.errors = nan(6, 1);    % Vector with errors from the different methods
    results.accuracy = nan(6, 1);
    Train_labels_vec = nan(size(Train_lab, 1), 1);
    for kk = 1: 1: size(Train_lab, 1)
        Train_labels_vec(kk, 1) = find(Train_lab(kk, :));
    end

    %% Least square regression with pseudo-inverse
    results.X1 = pinv(Train_dat) * Train_lab;
    B1_mod = Test_dat * results.X1;
    B1 = rowMax(B1_mod);
    results.errors(1) = norm(Test_lab - B1) / norm(Test_lab);
    assert(size(B1, 1) == size(Test_lab, 1) && size(B1, 2) == size(Test_lab, 2),...
        'Dimensions mismatch.')
    results.accuracy(1) = nnz(all(Test_lab == B1, 2)) / size(Test_lab, 1);

    %% QR decomposition with Matlab \ command
    results.X2 = Train_dat \ Train_lab;
    B2_mod = Test_dat * results.X2;
    B2 = rowMax(B2_mod);
    results.errors(2) = norm(Test_lab - B2) / norm(Test_lab);
    assert(size(B2, 1) == size(Test_lab, 1) && size(B2, 2) == size(Test_lab, 2),...
        'Dimensions mismatch.')
    results.accuracy(2) = nnz(all(Test_lab == B2, 2)) / size(Test_lab, 1);

    %% LASSO with L1 norm optimization
    results.X3 = nan(size(results.X2));
    results.lasso1_int = nan(1, 10);
    for ii = 1: 1: 10
        [results.X3(:, ii), stats] = lasso(Train_dat,...
            double(Train_labels_vec == ii), 'Lambda', lam1,...
            'Intercept', true);    % Note that 0 is relabeled as 10 already
        results.lasso1_int(ii) = stats.Intercept;
    end
    B3_mod = Test_dat * results.X3 + results.lasso1_int;
    B3 = rowMax(B3_mod);
    results.errors(3) = norm(Test_lab - B3) / norm(Test_lab);
    assert(size(B3, 1) == size(Test_lab, 1) && size(B3, 2) == size(Test_lab, 2),...
        'Dimensions mismatch.')
    results.accuracy(3) = nnz(all(Test_lab == B3, 2)) / size(Test_lab, 1);
    
    %% LASSO with L1 and L2 norm optimization
    results.X4 = nan(size(results.X2));
    results.lasso2_int = nan(1, 10);
    for ii = 1: 1: 10
        [results.X4(:, ii), stats] = lasso(Train_dat,...
            double(Train_labels_vec == ii), 'Lambda', lam1, 'Alpha', lam2,...
            'Intercept', true);    % Note that 0 is relabeled as 10 already
        results.lasso2_int(ii) = stats.Intercept;
    end
    B4_mod = Test_dat * results.X4 + results.lasso2_int;
    B4 = rowMax(B4_mod);
    results.errors(4) = norm(Test_lab - B4) / norm(Test_lab);
    assert(size(B4, 1) == size(Test_lab, 1) && size(B4, 2) == size(Test_lab, 2),...
        'Dimensions mismatch.')
    results.accuracy(4) = nnz(all(Test_lab == B4, 2)) / size(Test_lab, 1);
    
    %% Robustfit solver
    results.X5 = nan(size(results.X2));
    results.Robust_const = nan(1, 10);
    for ii = 1: 1: 10
        temp = robustfit(Train_dat, double(Train_labels_vec == ii), 'ols');
        results.X5(:, ii) = temp(2:end);
        results.Robust_const(ii) = temp(1);
    end
    B5_mod = Test_dat * results.X5 + results.Robust_const;
    B5 = rowMax(B5_mod);
    results.errors(5) = norm(Test_lab - B5) / norm(Test_lab);
    assert(size(B5, 1) == size(Test_lab, 1) && size(B5, 2) == size(Test_lab, 2),...
        'Dimensions mismatch.')
    results.accuracy(5) = nnz(all(Test_lab == B5, 2)) / size(Test_lab, 1);
    
    %% Ridge regression
    results.X6 = nan(size(results.X2));
    results.Ridge_const = nan(1, 10);
    for ii = 1: 1: 10
        temp = ridge(double(Train_labels_vec == ii), Train_dat, 0.5, 0);
        results.X6(:, ii) = temp(2:end);
        results.Ridge_const(ii) = temp(1);
    end
    B6_mod = Test_dat * results.X6 + results.Ridge_const;
    B6 = rowMax(B6_mod);
    results.errors(6) = norm(Test_lab - B6) / norm(Test_lab);
    assert(size(B6, 1) == size(Test_lab, 1) && size(B6, 2) == size(Test_lab, 2),...
        'Dimensions mismatch.')
    results.accuracy(6) = nnz(all(Test_lab == B6, 2)) / size(Test_lab, 1);
    
end