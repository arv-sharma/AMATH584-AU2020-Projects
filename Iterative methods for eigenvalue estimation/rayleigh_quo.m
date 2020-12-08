function [lambda_est, v, iterations] = rayleigh_quo(A, v_0, num_of_iter)

if size(A, 1) ~= size(A, 2)
    error('Input matrix is not square.')
end

m = size(A, 1);
v = v_0 / norm(v_0);
lambda_est = v' * A * v;

for iterations = 1: 1: num_of_iter
%     if cond(A - lambda_est * eye(m)) > 1e12
%         warning('Condition number over 10^12. Guess must be close to an eigenvalue.')
%         break;
%     end
%     w = inv(A - lambda_est * eye(m)) * v;
    w = (A - lambda_est * eye(m)) \ v;
    v = w / norm(w);
    lambda_est = v' * A * v;    
end

end
    