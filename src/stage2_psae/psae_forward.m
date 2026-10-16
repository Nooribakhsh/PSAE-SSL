function out = psae_forward(model, X)
% PSAE_FORWARD  Full forward pass: encode/decode through every branch,
%               concatenate + fuse the latent codes (Equation eq_branch),
%               classify the fused code (softmax over Equation eq_sup's
%               logits), and average the per-branch reconstructions
%               (the "fused reconstruction... averaged from branch
%               decoders" mentioned alongside Equation eq_rec).
%
%   X   : inputDim x N
%   out : struct with fields Hb_all, Xhat_all, branchCaches, Hcat, Hf,
%         fusionCache, probs, Xhat, X, N — everything PSAE_BACKWARD and
%         COMPUTE_LOSSES need.

B = model.B;
N = size(X, 2);

Hb_all = cell(B, 1);
Xhat_all = cell(B, 1);
branchCaches = cell(B, 1);
for b = 1:B
    % P2: class/feature specialization — each branch sees a masked view
    if isfield(model, 'branchInputMasks') && ~isempty(model.branchInputMasks{b})
        Xb = X .* model.branchInputMasks{b};  % broadcast inputDim x 1 over N
    else
        Xb = X;
    end
    [Hb, Xhat_b, caches] = branch_forward(model.branches{b}, Xb);
    Hb_all{b} = Hb;
    Xhat_all{b} = Xhat_b;
    branchCaches{b} = caches;
end

Hcat = cell2mat(Hb_all(:)); % sum(latentDims) x N, vertical concatenation of branch codes

[Hf, fusionCache] = dense_forward(model.fusion.W, model.fusion.b, Hcat, model.fusion.activation);

% Linear classifier + softmax (Equation eq_sup's P(class))
logits = model.clf.W * Hf + model.clf.b; % K x N
logits = logits - max(logits, [], 1);    % numerical stability
expL = exp(logits);
probs = expL ./ sum(expL, 1);

% Averaged reconstruction across branches
Xhat = zeros(size(X));
for b = 1:B
    Xhat = Xhat + Xhat_all{b};
end
Xhat = Xhat / B;

out.Hb_all = Hb_all;
out.Xhat_all = Xhat_all;
out.branchCaches = branchCaches;
out.Hcat = Hcat;
out.Hf = Hf;
out.fusionCache = fusionCache;
out.probs = probs;
out.Xhat = Xhat;
out.X = X;
out.N = N;
end
