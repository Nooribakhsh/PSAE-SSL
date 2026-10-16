function grads = branch_backward(branch, caches, dXhat_b, dHb_extra)
% BRANCH_BACKWARD  Backprop through one PSAE branch's decoder then
%                   encoder.
%
%   dXhat_b   : gradient of the (weighted) total loss w.r.t. this
%               branch's reconstruction output, coming from L_rec
%   dHb_extra : gradient of the (weighted) total loss w.r.t. this
%               branch's latent code Hb, coming from the fusion path
%               (L_sup + L_sep). Pass zeros(size(Hb)) during
%               unsupervised pretraining.
%
%   grads : struct with .enc, .dec cell arrays of struct('dW','db')

L = branch.L;
grads.enc = cell(L, 1);
grads.dec = cell(L, 1);

% ---- decoder: output -> latent ----
dA = dXhat_b;
for l = L:-1:1
    lyr = branch.dec{l};
    [dW, db, dA] = dense_backward(dA, caches.dec{l}, lyr.W);
    grads.dec{l} = struct('dW', dW, 'db', db);
end
% dA is now dL_rec/dHb (gradient reaching the latent code via reconstruction)
dHb_total = dA + dHb_extra;

% ---- encoder: latent -> input ----
dA = dHb_total;
for l = L:-1:1
    lyr = branch.enc{l};
    [dW, db, dA] = dense_backward(dA, caches.enc{l}, lyr.W);
    grads.enc{l} = struct('dW', dW, 'db', db);
end
% dA here would be dL/dX (gradient w.r.t. the fixed input data); discarded.
end
