function labels = label_guided_clustering(Hf, adjList, labeledIdx, labeledLabels, classList, baseLabels, seed)
% LABEL_GUIDED_CLUSTERING  Semi-supervised Stage-3 refinement.
%
%   Combines two label-driven scores for every node:
%     (a) prototype score  : softmax of cosine similarity between the node's
%                            fused embedding and the class prototypes
%                            (mean embedding of the labeled nodes of a class);
%     (b) propagation score: label propagation (Zhou et al.) over a sparse
%                            graph = original edges + embedding kNN edges.
%   score = beta * propagation + (1 - beta) * prototype.
%   beta is chosen on held-out LABELED nodes only (no ground truth of the
%   unlabeled nodes is used); with too few labels beta = 0.5.
%   Labeled nodes keep their given labels.
%
%   Hf          d x n fused embeddings (columns = nodes)
%   adjList     1 x n cell adjacency list
%   labeledIdx, labeledLabels : labeled node ids and their class ids
%   classList   sorted unique ground-truth class ids (defines K)
%   baseLabels  unsupervised partition (used only for classes that have no
%               labeled node, to build a prototype for them)
%   seed        random seed (hold-out split)
%
%   Output: 1 x n labels in 1..K (class index in classList order).

if nargin < 7 || isempty(seed), seed = 1; end
n = size(Hf, 2);
if size(Hf, 1) == n && size(Hf, 2) ~= n
    Hf = Hf';
end
colN = sqrt(sum(Hf.^2, 1)) + 1e-12;
Hf = Hf ./ colN;
K = numel(classList);
labeledIdx = labeledIdx(:)';
[~, lab] = ismember(labeledLabels(:)', classList(:)');
keep = lab > 0;
labeledIdx = labeledIdx(keep);
lab = lab(keep);
baseLabels = baseLabels(:)';

S = build_propagation_operator(Hf, adjList, n);

% ---- choose beta on held-out labeled nodes ----
betas = [0 0.25 0.5 0.75 1];
beta = 0.5;
rng(seed + 777);
hold = false(1, numel(labeledIdx));
for c = 1:K
    mem = find(lab == c);
    nh = floor(0.25 * numel(mem));
    if numel(mem) >= 2 && nh >= 1
        pick = mem(randperm(numel(mem), nh));
        hold(pick) = true;
    end
end
if sum(hold) >= K && sum(~hold) >= K
    [Fp, Pp] = label_scores(Hf, S, labeledIdx(~hold), lab(~hold), K, baseLabels);
    acc = zeros(size(betas));
    for bi = 1:numel(betas)
        sc = betas(bi) * Fp + (1 - betas(bi)) * Pp;
        [~, pred] = max(sc(labeledIdx(hold), :), [], 2);
        acc(bi) = mean(pred(:) == lab(hold)');
    end
    best = max(acc);
    cand = find(acc >= best - 1e-12);
    [~, j] = min(abs(betas(cand) - 0.5));   % tie-break toward 0.5
    beta = betas(cand(j));
end

% ---- final scores with all labeled nodes ----
[Fn, Pn] = label_scores(Hf, S, labeledIdx, lab, K, baseLabels);
score = beta * Fn + (1 - beta) * Pn;
[~, labels] = max(score, [], 2);
labels = labels(:)';
labels(labeledIdx) = lab;
end

% ------------------------------------------------------------------------
function [Fn, Pn] = label_scores(Hf, S, idx, lab, K, baseLabels)
n = size(Hf, 2);
d = size(Hf, 1);

% class prototypes
protos = zeros(d, K);
has = false(1, K);
for c = 1:K
    m = idx(lab == c);
    if ~isempty(m)
        protos(:, c) = mean(Hf(:, m), 2);
        has(c) = true;
    end
end
if ~all(has) && ~isempty(baseLabels) && numel(baseLabels) == n
    protos = fill_missing_protos(Hf, protos, has, idx, lab, K, baseLabels);
end
pn = sqrt(sum(protos.^2, 1)) + 1e-12;
protos = protos ./ pn;
Sim = Hf' * protos;                        % n x K cosine
tau = 0.1;
Z = Sim / tau;
Z = Z - max(Z, [], 2);
Pn = exp(Z);
Pn = Pn ./ (sum(Pn, 2) + 1e-12);

% label propagation
Y0 = sparse(idx(:), lab(:), 1, n, K);
Y0 = full(Y0);
F = Y0;
alpha = 0.9;
for it = 1:30
    F = alpha * (S * F) + (1 - alpha) * Y0;
end
Fn = F ./ (sum(F, 2) + 1e-12);
Fn(sum(F, 2) < 1e-12, :) = 1 / K;
end

% ------------------------------------------------------------------------
function protos = fill_missing_protos(Hf, protos, has, idx, lab, K, baseLabels)
% Prototypes for classes without labeled nodes come from unsupervised
% clusters that are not matched to any class that has labeled nodes.
bl = unique(baseLabels);
nb = numel(bl);
used = false(1, nb);
for c = find(has)
    m = idx(lab == c);
    cnt = zeros(1, nb);
    for j = 1:nb
        cnt(j) = sum(baseLabels(m) == bl(j));
    end
    cnt(used) = -1;
    [mx, j] = max(cnt);
    if mx > 0, used(j) = true; end
end
free = find(~used);
miss = find(~has);
for t = 1:numel(miss)
    if t > numel(free), break; end
    mem = find(baseLabels == bl(free(t)));
    if ~isempty(mem)
        protos(:, miss(t)) = mean(Hf(:, mem), 2);
    end
end
end

% ------------------------------------------------------------------------
function S = build_propagation_operator(Hf, adjList, n)
% Symmetric-normalized sparse operator of (edges + embedding kNN edges).
A = adjlist_to_sparse(adjList);
A = spones(max(A, A'));
A = A - spdiags(diag(A), 0, n, n);
Sk = knn_graph(Hf, 10);
W = A + Sk;
d = full(sum(W, 2));
d(d < 1e-12) = 1e-12;
Dm = spdiags(1 ./ sqrt(d), 0, n, n);
S = Dm * W * Dm;
S = (S + S') / 2;
end

% ------------------------------------------------------------------------
function Sk = knn_graph(Hf, k)
n = size(Hf, 2);
X = Hf';
k = min(k, n - 1);
chunk = max(1, min(n, floor(2e7 / n)));
I = zeros(n * k, 1); J = zeros(n * k, 1); V = zeros(n * k, 1);
pos = 0;
for s = 1:chunk:n
    e = min(s + chunk - 1, n);
    G = X(s:e, :) * X';
    rows = (s:e)';
    G(sub2ind(size(G), (1:numel(rows))', rows)) = -Inf;
    [vals, idx] = maxk(G, k, 2);
    cnt = numel(rows) * k;
    I(pos+1:pos+cnt) = repelem(rows, k);
    J(pos+1:pos+cnt) = reshape(idx', [], 1);
    V(pos+1:pos+cnt) = reshape(max(vals', 0), [], 1);
    pos = pos + cnt;
end
Sk = sparse(I(1:pos), J(1:pos), V(1:pos), n, n);
Sk = max(Sk, Sk');
end
