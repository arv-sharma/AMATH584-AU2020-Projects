function Data_mat = read_MNIST(filename)
    % Reads in MNIST data file and parses it using information from:
    % http://yann.lecun.com/exdb/mnist/
    
    fileID = fopen(filename, 'r', 'b');
    
    magic_num = fread(fileID, 1, 'int32', 'ieee-be');
    
    if magic_num == 2049
        fprintf('\nReading label file...\n')
        num_items = fread(fileID, 1, 'int32', 'ieee-be');
        Data_mat = fread(fileID, inf, 'uint8', 'ieee-be');
        fclose(fileID);
        assert(length(Data_mat) == num_items, '\nData read unsuccessful.\n')
    elseif magic_num == 2051
        fprintf('\nReading image file...\n')
        num_items = fread(fileID, 1, 'int32', 'ieee-be');
        num_rows = fread(fileID, 1, 'int32', 'ieee-be');
        num_cols = fread(fileID, 1, 'int32', 'ieee-be');
        temp = fread(fileID, [num_rows * num_cols, inf], 'uint8', 'ieee-be');
        fclose(fileID);
        temp = reshape(temp, [num_rows, num_cols, num_items]);
        temp = permute(temp, [2, 1, 3]);    % To transpose the images into a
                                            % form that is native to
                                            % Matlab.
        Data_mat = 255 - reshape(temp, [num_rows * num_cols,...
            num_items]);    % Subtract 255 to make pixels of interest black
        assert(size(Data_mat, 2) == num_items, '\nData read unsuccessful.\n')
    else
        error('\nMagic number not valid. Check input file.\n')
    end
    fprintf('\n... successful.\n')
end
    