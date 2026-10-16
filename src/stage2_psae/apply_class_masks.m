function model = apply_class_masks(model, X, labeledIdx, labeledLabels)
% APPLY_CLASS_MASKS  Specialize branch input masks using labeled nodes.
%
%   For each class k, rank features by a simple discriminability score
%   (between-class vs within-class variance on labeled columns of X).
%   Branch b associated with class model.branchClassId(b) keeps the top
%   features for that class (plus a small shared base), matching the
%   paper's class-specific mask design more closely than pure random masks.
%
%   X             : d x n embeddings (Stage-1)
%   labeledIdx    : indices of labeled nodes
%   labeledLabels : labels in {1..K}

if isempty(labeledIdx) || numel(unique(labeledLabels)) < 2
    return
end

d = size(X, 1);
K = model.numClasses;
B = model.B;
Xl = X(:, labeledIdx);
yl = labeledLabels(:)';

% Global mean / var
muG = mean(Xl, 2);
varG = var(Xl, 0, 2) + 1e-8;

% Per-class mean
score = zeros(d, K);  % higher = more discriminative for class k
for k = 1:K
    idx = (yl == k);
    if nnz(idx) < 2
        score(:, k) = varG;
        continue
    end
    muK = mean(Xl(:, idx), 2);
    varK = var(Xl(:, idx), 0, 2) + 1e-8;
    % Fisher-like: separation from global / within-class scatter
    score(:, k) = ((muK - muG).^2) ./ varK;
end

keepFrac = 0.55;  % each branch keeps ~55% features
nKeep = max(1, round(keepFrac * d));
sharedN = max(1, round(0.15 * d));
[~, sharedOrd] = sort(varG, 'descend');
shared = sharedOrd(1:sharedN);

for b = 1:B
    k = model.branchClassId(b);
    [~, ord] = sort(score(:, k), 'descend');
    keep = unique([ord(1:nKeep); shared]);
    mask = zeros(d, 1);
    mask(keep) = 1;
    if ~any(mask), mask(1) = 1; end
    model.branchInputMasks{b} = mask;
end

% Coverage guarantee
covered = false(d, 1);
for b = 1:B
    covered = covered | (model.branchInputMasks{b} > 0);
end
missing = find(~covered);
for ii = 1:numel(missing)
    b = mod(ii - 1, B) + 1;
    model.branchInputMasks{b}(missing(ii)) = 1;
end
end
