function [Z0, history] = train_skipgram(walks, vocabSize, opts)
% TRAIN_SKIPGRAM  Skip-gram with negative sampling (word2vec-style SGD).
%
%   Same model and objective as before: for every (center u, context v)
%   pair, one positive and Kneg negative targets are drawn and W_in(u,:)
%   and W_out(targets,:) take one logistic-loss gradient step. The gradient
%   for the center row is accumulated over the 1+Kneg targets and applied
%   once (as in the reference word2vec implementation).
%
%   Implementation notes (speed only, no change of the objective):
%     * pairs are built once with preallocated, vectorised code;
%     * negatives are sampled in vectorised chunks from the alias table;
%     * the SGD loop updates W_in / W_out in place (no function calls that
%       would copy the full matrices on every update).

d          = opts.d;
w          = opts.w;
Kneg       = opts.numNegative;
eta0       = opts.learningRate;
numEpochs  = opts.numEpochs;

% Small random init (word2vec style)
W_in  = (rand(vocabSize, d) - 0.5) / d;
W_out = (rand(vocabSize, d) - 0.5) / d;

% Unigram^0.75 negative-sampling distribution
allNodes = [];
for i = 1:numel(walks)
    allNodes = [allNodes; walks{i}(:)]; %#ok<AGROW>
end
freq = accumarray(allNodes, 1, [vocabSize 1]);
freq(freq == 0) = 1;
unigram = freq .^ 0.75;
unigram = unigram / sum(unigram);
[negJ, negQ] = alias_setup(unigram);

% Positive (center, context) pairs, built once. For every walk and every
% offset 1..w both (a,b) and (b,a) are generated, i.e. the same multiset
% of pairs as scanning a +/-w window around every position.
cen = cell(numel(walks), 1);
ctx = cell(numel(walks), 1);
for i = 1:numel(walks)
    wk = walks{i}(:);
    L = numel(wk);
    ca = cell(min(w, max(L - 1, 0)), 1);
    cb = cell(numel(ca), 1);
    for off = 1:numel(ca)
        a = wk(1:L - off);
        b = wk(1 + off:L);
        ca{off} = [a; b];
        cb{off} = [b; a];
    end
    if ~isempty(ca)
        cen{i} = vertcat(ca{:});
        ctx{i} = vertcat(cb{:});
    end
end
cenAll = vertcat(cen{:});
ctxAll = vertcat(ctx{:});
clear cen ctx
nPairs = numel(cenAll);
fprintf('  [Skip-gram] %d positive pairs, %d epochs, lr=%.3f\n', nPairs, numEpochs, eta0);

history = struct('epoch', {}, 'avgLoss', {});
chunk  = 200000;                    % pairs processed per vectorised block
lab    = [1; zeros(Kneg, 1)];       % targets: 1 positive, Kneg negatives
verbose = nPairs > 1e6;             % print every epoch on large corpora

for epoch = 1:numEpochs
    tEpoch = tic;
    eta = eta0 * max(0.0001, 1 - (epoch - 1) / numEpochs);
    totalLoss = 0;
    ord = randperm(nPairs);

    for c0 = 1:chunk:nPairs
        c1 = min(c0 + chunk - 1, nPairs);
        sel = ord(c0:c1);
        U = cenAll(sel);
        T = [ctxAll(sel), draw_negatives(negJ, negQ, U, Kneg)];   % m x (1+Kneg)
        m = numel(U);

        for t = 1:m
            u  = U(t);
            tg = T(t, :);
            win = W_in(u, :);
            Wo  = W_out(tg, :);                 % (1+Kneg) x d
            s = Wo * win';
            s = max(min(s, 6), -6);
            p = 1 ./ (1 + exp(-s));
            g = p - lab;                        % dL/dscore
            totalLoss = totalLoss - log(max(p(1), 1e-10)) ...
                - sum(log(max(1 - p(2:end), 1e-10)));
            W_in(u, :)   = win - eta * (g' * Wo);
            W_out(tg, :) = Wo  - eta * (g * win);
        end
    end

    avgLoss = totalLoss / max(nPairs * (1 + Kneg), 1);
    history(end + 1) = struct('epoch', epoch, 'avgLoss', avgLoss); %#ok<AGROW>
    if verbose || epoch == 1 || mod(epoch, max(1, floor(numEpochs/5))) == 0 || epoch == numEpochs
        fprintf('  [Skip-gram] epoch %d/%d - avg loss = %.4f (eta=%.4f) [%.1fs]\n', ...
            epoch, numEpochs, avgLoss, eta, toc(tEpoch));
    end
end

% Row-normalize embeddings (common for downstream clustering)
Z0 = W_in;
nrm = sqrt(sum(Z0 .^ 2, 2)) + 1e-12;
Z0 = Z0 ./ nrm;
end

function neg = draw_negatives(J, q, U, Kneg)
% Vectorised alias-table draws; resample entries equal to the center node
% (up to 20 attempts, as in the previous implementation).
m = numel(U);
neg = alias_draw_vec(J, q, m * Kneg);
neg = reshape(neg, m, Kneg);
Um = repmat(U(:), 1, Kneg);
bad = (neg == Um);
attempts = 0;
while any(bad(:)) && attempts < 20
    neg(bad) = alias_draw_vec(J, q, nnz(bad));
    bad = (neg == Um);
    attempts = attempts + 1;
end
end

function idx = alias_draw_vec(J, q, m)
K  = numel(J);
kk = floor(rand(m, 1) * K) + 1;
acc = rand(m, 1) < q(kk);
idx = kk;
idx(~acc) = J(kk(~acc));
end
