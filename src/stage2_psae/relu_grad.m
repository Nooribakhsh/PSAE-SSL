function dZ = relu_grad(dA, Z)
% RELU_GRAD  Backprop through ReLU: dZ = dA .* 1[Z>0]
dZ = dA .* (Z > 0);
end
