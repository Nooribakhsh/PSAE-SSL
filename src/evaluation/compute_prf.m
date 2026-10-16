function [precision, recall, f1] = compute_prf(trueLabels, predLabels, average)
% COMPUTE_PRF  Precision / Recall / F1 after majority-vote cluster→class map.
%
%   average: 'macro' (default) or 'micro'
%   Tables report macro-averaged P/R (hence P can differ from R).

if nargin < 3 || isempty(average)
    average = 'macro';
end

trueLabels = trueLabels(:);
predLabels = predLabels(:);
[uy, ~, iy] = unique(trueLabels);
[uc, ~, ic] = unique(predLabels);
Ky = numel(uy);
Kc = numel(uc);

cont = accumarray([ic, iy], 1, [Kc, Ky]); % rows=pred clusters, cols=true classes
[~, mapTo] = max(cont, [], 2);           % majority class per predicted cluster
mapped = mapTo(ic);                      % predicted class id in 1..Ky

eps_ = 1e-12;
if strcmpi(average, 'micro')
    % Global counts
    TP = sum(mapped == iy);
    % For multi-class micro-P/R they coincide when every node is assigned
    precision = TP / (numel(iy) + eps_);
    recall    = precision;
    f1        = precision;
    return
end

% Macro: unweighted mean over true classes
Plist = zeros(Ky, 1);
Rlist = zeros(Ky, 1);
for k = 1:Ky
    predPos = (mapped == k);
    truePos = (iy == k);
    TP = sum(predPos & truePos);
    FP = sum(predPos & ~truePos);
    FN = sum(~predPos & truePos);
    Plist(k) = TP / (TP + FP + eps_);
    Rlist(k) = TP / (TP + FN + eps_);
end
precision = mean(Plist);
recall    = mean(Rlist);
f1        = 2 * precision * recall / (precision + recall + eps_);
end
