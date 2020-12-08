clearvars
close all

%% Eigenvalues, eigenvectors of random, symmetric matrix

m = 10;
B = rand(m);
B_sign = randi([0, 1], [m, m]);
B_sign(B_sign == 0) = -1;
B = B .* B_sign;
A = triu(B) + triu(B)' - diag(diag(B)); % creating a symmetric matrix from
                                        % random matrix B
[V, D] = eigs(A, 10);

fig = figure;
fig.Units = 'inches';
fig.Position = [-.1 1.8 6 4.5];
fig.PaperUnits = 'inches';
fig.PaperSize = [6 4.5];
h = plot(real(diag(D)), imag(diag(D)), 'k.', 'MarkerSize', 12);
hold on


%% Power iteration to find eigenvalues, eigenvectors of random, symmetrix matrix

num_of_iter = round(2 .^ (0: 0.5: 20));
eig1_est = nan(length(num_of_iter), 1);
eig1_sym = D(1);

for ii = 1: 1: length(num_of_iter)  
    eig1_est(ii) = power_iter(A, num_of_iter(ii), 1);
end

eig1_est_err = abs(eig1_sym - eig1_est);

fig1 = figure;
fig1.Units = 'inches';
fig1.Position = [-.1 1.8 6 4.5];
fig1.PaperUnits = 'inches';
fig1.PaperSize = [6 4.5];
ax = gca;
h1 = plot(num_of_iter, eig1_est_err, 'k.');
h1.MarkerSize = 12;
ax.XScale = 'log';
ax.YScale = 'log';
hold on

%% Rayleigh quotient iteration to find eigenvalues, eigenvectors of random,
% symmetrix matrix

eig_vals = diag(D);
eig_vecs = V;
labels = cell(m, 1);
count = 0;
raw_count = 0;

fig2 = figure;
fig2.Units = 'inches';
fig2.Position = [-.1 1.8 6 4.5];
fig2.PaperUnits = 'inches';
fig2.PaperSize = [6 4.5];
hold on
fig3 = figure;
fig3.Units = 'inches';
fig3.Position = [-.1 1.8 6 4.5];
fig3.PaperUnits = 'inches';
fig3.PaperSize = [6 4.5];
hold on

colorVec = hsv(m);
lineStyle = {'-', '-.', '--', ':', '-', '-.', ':', '--', '-', '-.'};

while ~isempty(eig_vals)
    
    raw_count = raw_count + 1;
    max_iters = 20;
    v_guess = rand(m, 1);
    v_guess_sign_re = randi([0, 1], [m, 1]);
    v_guess_sign_re(v_guess_sign_re == 0) = -1;
    v_guess = v_guess .* v_guess_sign_re;
    
    lambda_est = rayleigh_quo(A, v_guess, max_iters);
    
    if any(abs(eig_vals - lambda_est) < 1e-8)
        count = count + 1;
        id = find(abs(eig_vals - lambda_est) < 1e-8);
        eig_val = eig_vals(id);
        eig_vec = eig_vecs(:, id);
        eig_vals(id) = [];
        eig_vecs(:, id) = [];
        labels{count} = strcat('\lambda=', num2str(lambda_est));
        
        eigval_est_err = nan(max_iters, 1);
        eigvec_est_err = nan(max_iters, 1);
        
        for ii = 1: 1: max_iters
            [lambda_est, v, ~] = rayleigh_quo(A, v_guess, ii);
            eigval_est_err(ii) = abs(eig_val - lambda_est);
            eigvec_est_err(ii) = min([norm(eig_vec - v),...
                norm(-eig_vec - v)]);    % To account for the fact that
                                        % eigenvectors could be negative of
                               
                                        % true value
        end
        
        figure(fig2);
        plot(eigval_est_err, 'Marker', '.', 'MarkerSize', 12, 'Color',...
            colorVec(count, :), 'LineStyle', lineStyle{count},...
            'LineWidth', 1.1)
        
        figure(fig3);
        plot(eigvec_est_err, 'Marker', '.', 'MarkerSize', 12, 'Color',...
            colorVec(count, :), 'LineStyle', lineStyle{count},...
            'LineWidth', 1.1)
    end
    
end

fprintf('\nWith A being a symmetric %d x %d random matrix:\n', m, m)
fprintf('\nTotal number of initial guesses = %d.\n', raw_count)
fprintf('\nTotal number of useful guesses = %d.\n', m)
fprintf('\nNeeded %.2f (on average) guesses for each eigenvector.\n',...
    raw_count / m)

figure(fig2);
ax2 = gca;
ax2.Box = 'on';
ax2.YScale = 'log';
legend(labels)
xlabel('Number of iterations')
ylabel('Error, |\lambda_{est} - \lambda|')
% savefig(fig2, 'sym_eigval_errors')
% print('sym_eigval_errors', '-depsc', '-r300')

figure(fig3);
ax3 = gca;
ax3.Box = 'on';
ax3.YScale = 'log';
legend(labels)
xlabel('Number of iterations')
ylabel('$\|${\boldmath${x_{est}}$ - \boldmath${x}$}$\|$', 'Interpreter',...
    'latex')
% savefig(fig3, 'sym_eigvec_errors')
% print('sym_eigvec_errors', '-depsc', '-r300')

%% Eigenvalues, eigenvectors of random, asymmetric matrix

m = 10;
A = rand(m);
A_sign = randi([0, 1], [m, m]);
A_sign(A_sign == 0) = -1;
A = A .* A_sign;
[V, D] = eigs(A, 10);

figure(fig);
ha = plot(diag(D), 'r*', 'MarkerSize', 6);
legend('Symmetric', 'Asymmetric')
xlabel('Re(\lambda)')
ylabel('Im(\lambda)')
% savefig(fig, 'eig_values')
% print('eig_values', '-depsc', '-r300')

%% Power iteration to find eigenvalues, eigenvectors of random, asymmetrix matrix

num_of_iter = round(2 .^ (0: 0.5: 20));
eig1_est = nan(length(num_of_iter), 1);
eig1_asym = D(1);

for ii = 1: 1: length(num_of_iter)  
    eig1_est(ii) = power_iter(A, num_of_iter(ii), 0);
end

eig1_est_err = abs(eig1_asym - eig1_est);

figure(fig1);
h1a = plot(num_of_iter, eig1_est_err, 'r*', 'MarkerSize', 6);
legend(['Symmetric (\lambda_1 = ', num2str(eig1_sym), ')'],...
    ['Asymmetric (\lambda_1 = ', num2str(eig1_asym), ')'])
xlabel('Number of iterations')
ylabel('Error, |\lambda_{1, est} - \lambda_1|')
% savefig(fig1, 'sym_domeigval_errors')
% print('sym_domeigval_errors', '-depsc', '-r300')

%% Rayleigh quotient iteration to find eigenvalues, eigenvectors of random,
% asymmetrix matrix

eig_vals = diag(D);
eig_vecs = V;
labels = cell(m, 1);
count = 0;
raw_count1 = 0;

fig4 = figure;
fig4.Units = 'inches';
fig4.Position = [-.1 1.8 6 4.5];
fig4.PaperUnits = 'inches';
fig4.PaperSize = [6 4.5];
hold on
fig5 = figure;
fig5.Units = 'inches';
fig5.Position = [-.1 1.8 6 4.5];
fig5.PaperUnits = 'inches';
fig5.PaperSize = [6 4.5];
hold on

colorVec = hsv(m);
lineStyle = {'-', '-.', '--', ':', '-', '-.', ':', '--', '-', '-.'};

while ~isempty(eig_vals)
    
    raw_count1 = raw_count1 + 1;
    max_iters = 50;
    v_guess_sign_re = randi([0, 1], [m, 1]);
    v_guess_sign_re(v_guess_sign_re == 0) = -1;
    v_guess_sign_im = randi([0, 1], [m, 1]);
    v_guess_sign_im(v_guess_sign_im == 0) = -1;
    v_guess = rand(m, 1) .* v_guess_sign_re + 1i * rand(m, 1) .*...
        v_guess_sign_im;
    
    lambda_est = rayleigh_quo(A, v_guess, max_iters);
    
    if any(abs(eig_vals - lambda_est) < 1e-8)
        count = count + 1;
        id = find(abs(eig_vals - lambda_est) < 1e-8);
        eig_val = eig_vals(id);
        eig_vec = eig_vecs(:, id);
        eig_vals(id) = [];
        eig_vecs(:, id) = [];
        labels{count} = strcat('\lambda=', num2str(lambda_est));
        
        eigval_est_err = nan(max_iters, 1);
        eigvec_est_err = nan(max_iters, 1);
        
        for ii = 1: 1: max_iters
            [lambda_est, v, ~] = rayleigh_quo(A, v_guess, ii);
            eigval_est_err(ii) = abs(eig_val - lambda_est);
            eigvec_est_err(ii) = min([norm(eig_vec - v),...
                norm(-eig_vec - v)]);    % To account for the fact that
                                        % eigenvectors could be negative of
                               
                                        % true value
        end
        
        figure(fig4);
        plot(eigval_est_err, 'Marker', '.', 'MarkerSize', 12, 'Color',...
            colorVec(count, :), 'LineStyle', lineStyle{count},...
            'LineWidth', 1.1)
        
        figure(fig5);
        plot(eigvec_est_err, 'Marker', '.', 'MarkerSize', 12, 'Color',...
            colorVec(count, :), 'LineStyle', lineStyle{count},...
            'LineWidth', 1.1)
    end
    
    if raw_count1 > raw_count * m ^ 2 || raw_count1 > 20e3
        fprintf('Eigenvalues failed to coverge after %d guesses. Aborting.\n',...
            raw_count1)
        break
    end
    warning('off','last')
end

fprintf('\nWith A being an asymmetric %d x %d random matrix:\n', m, m)
fprintf('\nTotal number of initial guesses = %d.\n', raw_count1)
fprintf('\nTotal number of useful guesses = %d.\n', count)
fprintf('\nAlgorithm is able to find all eigenvalues.\n')
fprintf('\nThere is significant error in finding eigenvectors, however.\n')

figure(fig4);
ax4 = gca;
ax4.Box = 'on';
ax4.YScale = 'log';
legend(labels(1:count))
xlabel('Number of iterations')
ylabel('Error, |\lambda_{est} - \lambda|')
% savefig(fig4, 'asym_eigval_errors')
% print('asym_eigval_errors', '-depsc', '-r300')

figure(fig5);
ax5 = gca;
ax5.Box = 'on';
ax5.YScale = 'log';
legend(labels(1:count))
xlabel('Number of iterations')
ylabel('$\|${\boldmath${x_{est}}$ - \boldmath${x}$}$\|$', 'Interpreter',...
    'latex')
% savefig(fig5, 'asym_eigvec_errors')
% print('asym_eigvec_errors', '-depsc', '-r300')
