function [Z0, info] = node2vec(adjList, opts)
% NODE2VEC  Stage 1 of PSAE-SSL: learn topology-aware node
%               embeddings Z^(0) via biased random walks and Skip-gram
%               training. Implements Algorithm 1 of the paper end to
%               end.
%
%   [Z0, info] = NODE2VEC(adjList, opts)
%
%   Inputs
%     adjList : n x 1 cell array from LOAD_GRAPH
%     opts    : struct, all fields optional (defaults noted below).
%               The unbiased setting p=1, q=1 is the most stable choice
%               after exploring {0.25,0.5,1,2,4} for both (Section
%               "Implementation details"); d, l, r, w are configurable
%               per-dataset hyperparameters.
%
%       .d            embedding dimension                 (default 128)
%       .l            walk length                         (default 80)
%       .r            walks per node                       (default 10)
%       .p            return parameter                      (default 1)
%       .q            in-out parameter                       (default 1)
%       .w            Skip-gram context window radius       (default 10)
%       .numNegative  negative samples per pair               (default 5)
%       .numEpochs    Skip-gram training epochs                (default 5)
%       .learningRate Skip-gram initial learning rate      (default 0.025)
%       .seed         RNG seed for reproducibility               (default 1)
%
%   Outputs
%     Z0   : n x d embedding matrix (Z^(0) in the paper's notation)
%     info : struct with the walk corpus size and Skip-gram loss history,
%            useful for logging/plots in the experiments/ scripts

if nargin < 2
    opts = struct();
end
opts = set_default(opts, 'd', 128);
opts = set_default(opts, 'l', 80);
opts = set_default(opts, 'r', 10);
opts = set_default(opts, 'p', 1);
opts = set_default(opts, 'q', 1);
opts = set_default(opts, 'w', 10);
opts = set_default(opts, 'numNegative', 5);
opts = set_default(opts, 'numEpochs', 5);
opts = set_default(opts, 'learningRate', 0.025);
opts = set_default(opts, 'seed', 1);

rng(opts.seed);
n = numel(adjList);

fprintf('[Stage1] Building 2nd-order alias tables (p=%.3g, q=%.3g)...\n', opts.p, opts.q);
G = build_transition_probs(adjList, opts.p, opts.q);

fprintf('[Stage1] Simulating %d walks/node of length %d (n=%d nodes)...\n', opts.r, opts.l, n);
walks = cell(n * opts.r, 1);
cnt = 0;
for rep = 1:opts.r
    order = randperm(n); % iterate nodes in random order each pass, as in the reference implementation
    for ii = 1:n
        v = order(ii);
        cnt = cnt + 1;
        walks{cnt} = node2vec_walk(G, v, opts.l);
    end
end
walks = walks(1:cnt);

fprintf('[Stage1] Training Skip-gram (d=%d, window=%d, negatives=%d, epochs=%d)...\n', ...
    opts.d, opts.w, opts.numNegative, opts.numEpochs);
[Z0, history] = train_skipgram(walks, n, opts);

info.numWalks = numel(walks);
info.history  = history;
info.opts     = opts;

fprintf('[Stage1] Done. Z0 is %dx%d.\n', size(Z0, 1), size(Z0, 2));
end

function opts = set_default(opts, field, val)
if ~isfield(opts, field) || isempty(opts.(field))
    opts.(field) = val;
end
end
