function dZ = sigmoid_grad(dA, A)
% SIGMOID_GRAD  Backprop through Sigmoid given its OUTPUT A = sigmoid(Z):
% dZ = dA .* A .* (1 - A)
dZ = dA .* A .* (1 - A);
end
