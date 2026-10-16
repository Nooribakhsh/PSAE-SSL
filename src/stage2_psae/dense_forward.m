function [A, cache] = dense_forward(W, b, X, activation)
% DENSE_FORWARD  One fully-connected layer: Z = W*X + b, A = act(Z).
%
%   X : fan_in  x N (columns = samples)
%   W : fan_out x fan_in
%   b : fan_out x 1
%   activation : 'relu' | 'sigmoid' | 'linear'

Z = W * X + b;
switch activation
    case 'relu'
        A = relu(Z);
    case 'sigmoid'
        A = sigmoid(Z);
    case 'linear'
        A = Z;
    otherwise
        error('dense_forward:badActivation', 'Unknown activation "%s".', activation);
end

cache = struct('X', X, 'Z', Z, 'A', A, 'activation', activation);
end
