function [labeledIdx, labeledLabels] = sample_labeled_set(allLabels, labelFrac, seed)
% SAMPLE_LABELED_SET  "For the semi-supervised experiments, 10% of the
%                      nodes in each dataset were designated as
%                      ground-truth labeled data... The labeled nodes
%                      were selected via uniform random sampling over
%                      the full node set, without explicitly enforcing a
%                      minimum number of labeled nodes per community."
%                      (Section "Implementation details")
%
%   [labeledIdx, labeledLabels] = SAMPLE_LABELED_SET(allLabels, labelFrac, seed)
%
%   allLabels : 1 x n ground-truth community labels for every node
%   labelFrac : fraction of nodes to reveal as labeled (paper: 0.10)
%   seed      : RNG seed for reproducibility

if nargin < 2 || isempty(labelFrac)
    labelFrac = 0.10;
end
if nargin < 3
    seed = 1;
end
rng(seed);

n = numel(allLabels);
numLabeled = max(1, round(labelFrac * n));
perm = randperm(n);
labeledIdx = sort(perm(1:numLabeled));
labeledLabels = allLabels(labeledIdx);
end
