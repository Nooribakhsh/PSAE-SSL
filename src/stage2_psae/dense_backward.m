function [dW, db, dX] = dense_backward(dA, cache, W)
% DENSE_BACKWARD  Backprop through one fully-connected layer.
%
% Convention used throughout the PSAE code: every upstream gradient dA
% passed into this function is dL/dA where L is already a MEAN over the
% relevant sample set (e.g. dXhat = (2/N)*(Xhat-X) for the mean-squared
% reconstruction loss, or dLogits = (probs-onehot)/|L| for the
% mean cross-entropy loss). Because that normalization is baked into
% each per-sample column of dA, summing dZ*X' over the batch below
% already yields the correctly averaged parameter gradient — no further
% division by N is needed here, and this convention composes correctly
% across arbitrarily many chained dense_backward calls.

switch cache.activation
    case 'relu'
        dZ = relu_grad(dA, cache.Z);
    case 'sigmoid'
        dZ = sigmoid_grad(dA, cache.A);
    case 'linear'
        dZ = dA;
    otherwise
        error('dense_backward:badActivation', 'Unknown activation "%s".', cache.activation);
end

dW = dZ * cache.X';
db = sum(dZ, 2);
dX = W' * dZ;
end
