function results = compAxBSolversDigits(Train_dat, Train_lab,...
    Test_dat, Test_lab, lam1, lam2)

    % Comparison of 6 different methods to solve Ax = B
    
    results = struct;
    results.errors = nan(6, 1);    % Vector with errors from the different methods
    results.accuracy = nan(6, 1);

    %% Least square regression with pseudo-inverse
    results.X1 = pinv(Train_dat) * Train_lab;
    B1_mod = Test_dat * results.X1;
    B1 = B1_mod;
    B1(B1 < 0.5) = 0;
    B1(B1 >= 0.5) = 1;
    results.errors(1) = norm(Test_lab - B1) / norm(Test_lab);
    assert(size(B1, 1) == size(Test_lab, 1), 'Dimensions mismatch.')
    results.accuracy(1) = nnz(Test_lab == B1) / size(Test_lab, 1);

    %% QR decomposition with Matlab \ command
    results.X2 = Train_dat \ Train_lab;
    B2_mod = Test_dat * results.X2;
    B2 = B2_mod;
    B2(B2 < 0.5) = 0;
    B2(B2 >= 0.5) = 1;
    results.errors(2) = norm(Test_lab - B2) / norm(Test_lab);
    assert(size(B2, 1) == size(Test_lab, 1), 'Dimensions mismatch.')
    results.accuracy(2) = nnz(Test_lab == B2) / size(Test_lab, 1);

    %% LASSO with L1 norm optimization

   [results.X3, stats] = lasso(Train_dat, Train_lab, 'Lambda', lam1,...
            'Intercept', true);    % Note that 0 is relabeled as 10 already
    results.lasso1_int = stats.Intercept;
    B3_mod = Test_dat * results.X3 + results.lasso1_int;
    B3 = B3_mod;
    B3(B3 < 0.5) = 0;
    B3(B3 >= 0.5) = 1;
    results.errors(3) = norm(Test_lab - B3) / norm(Test_lab);
    assert(size(B3, 1) == size(Test_lab, 1), 'Dimensions mismatch.')
    results.accuracy(3) = nnz(Test_lab == B3) / size(Test_lab, 1);
    
    %% LASSO with L1 and L2 norm optimization

    [results.X4, stats] = lasso(Train_dat, Train_lab, 'Lambda', lam1,...
        'Alpha', lam2, 'Intercept', true);    % Note that 0 is relabeled as 10 already
    results.lasso2_int = stats.Intercept;
    B4_mod = Test_dat * results.X4 + results.lasso2_int;
    B4 = B4_mod;
    B4(B4 < 0.5) = 0;
    B4(B4 >= 0.5) = 1;
    results.errors(4) = norm(Test_lab - B4) / norm(Test_lab);
    assert(size(B4, 1) == size(Test_lab, 1), 'Dimensions mismatch.')
    results.accuracy(4) = nnz(Test_lab == B4) / size(Test_lab, 1);
    
    %% Robustfit solver

    temp = robustfit(Train_dat, Train_lab, 'ols');
    results.X5 = temp(2:end);
    results.Robust_const = temp(1);
    B5_mod = Test_dat * results.X5 + results.Robust_const;
    B5 = B5_mod;
    B5(B5 < 0.5) = 0;
    B5(B5 >= 0.5) = 1;
    results.errors(5) = norm(Test_lab - B5) / norm(Test_lab);
    assert(size(B5, 1) == size(Test_lab, 1), 'Dimensions mismatch.')
    results.accuracy(5) = nnz(Test_lab == B5) / size(Test_lab, 1);
    
    %% Ridge regression

    temp = ridge(Train_lab, Train_dat, 0.5, 0);
    results.X6 = temp(2:end);
    results.Ridge_const = temp(1);
    B6_mod = Test_dat * results.X6 + results.Ridge_const;
    B6 = B6_mod;
    B6(B6 < 0.5) = 0;
    B6(B6 >= 0.5) = 1;
    results.errors(6) = norm(Test_lab - B6) / norm(Test_lab);
    assert(size(B6, 1) == size(Test_lab, 1), 'Dimensions mismatch.')
    results.accuracy(6) = nnz(Test_lab == B6) / size(Test_lab, 1);
    
end