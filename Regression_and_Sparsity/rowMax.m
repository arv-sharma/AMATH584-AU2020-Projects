function output = rowMax(input)
    output = zeros(size(input));
    for ii = 1: 1: size(input, 1)
        temp = input(ii, :);
        [~, id] = max(temp);
        output(ii, id) = 1;
    end
    
end
        