function [lambda1_est, v] = power_iter(A, num_of_iter, symmetric)

    if size(A, 1) ~= size(A, 2)
        error('Input matrix is not square.')
    end

    m = size(A, 1);
    if symmetric
        v = rand(m, 1);
    else
        v = rand(m, 1) + 1i * rand(m, 1);
    end
    v = v / norm(v);    % Unit length vector

    for ii = 1: 1: num_of_iter
        w = A * v;
        v = w / norm(w);
    end
    lambda1_est = v' * A * v;
end


