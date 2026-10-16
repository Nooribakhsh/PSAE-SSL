function [Hb, Xhat_b, caches] = branch_forward(branch, X)
% BRANCH_FORWARD  Encode then decode X through one PSAE branch
%                 (Algorithm 2's per-branch encode/decode step).
%
%   X : inputDim x N
%
%   Hb     : latentDim x N, this branch's latent code h_{i,b}^{(L)}
%   Xhat_b : inputDim x N, this branch's reconstruction
%   caches : struct with .enc, .dec cell arrays of per-layer caches,
%            needed by BRANCH_BACKWARD

A = X;
encCaches = cell(branch.L, 1);
for l = 1:branch.L
    lyr = branch.enc{l};
    [A, c] = dense_forward(lyr.W, lyr.b, A, lyr.activation);
    encCaches{l} = c;
end
Hb = A;

decCaches = cell(branch.L, 1);
for l = 1:branch.L
    lyr = branch.dec{l};
    [A, c] = dense_forward(lyr.W, lyr.b, A, lyr.activation);
    decCaches{l} = c;
end
Xhat_b = A;

caches.enc = encCaches;
caches.dec = decCaches;
end
