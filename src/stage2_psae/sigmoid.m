function A = sigmoid(Z)
% SIGMOID  Equation eq_sigmoid: 1 / (1 + exp(-x))
A = 1 ./ (1 + exp(-Z));
end
