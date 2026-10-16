function [W, b, mW, vW, mb, vb] = adam_step(W, b, dW, db, mW, vW, mb, vb, t, lr, beta1, beta2, epsAdam, weightDecay)
% ADAM_STEP  One Adam update (Equation eq_adam) for a single weight
%            matrix / bias vector pair, with an added L2 weight-decay
%            term (Section 4.4: weight decay = 1e-4).
%
%   W <- W - eta * ghat_Adam( grad_W L )

dW = dW + weightDecay * W;

mW = beta1 * mW + (1 - beta1) * dW;
vW = beta2 * vW + (1 - beta2) * (dW .^ 2);
mWhat = mW / (1 - beta1 ^ t);
vWhat = vW / (1 - beta2 ^ t);
W = W - lr * mWhat ./ (sqrt(vWhat) + epsAdam);

mb = beta1 * mb + (1 - beta1) * db;
vb = beta2 * vb + (1 - beta2) * (db .^ 2);
mbHat = mb / (1 - beta1 ^ t);
vbHat = vb / (1 - beta2 ^ t);
b = b - lr * mbHat ./ (sqrt(vbHat) + epsAdam);
end
