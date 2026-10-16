function [labeledIdx, labeledLabels] = sample_labeled_set_stratified(allLabels, labelFrac, seed)
% SAMPLE_LABELED_SET_STRATIFIED  Stratified labeled-node sampling with a
%                                 guaranteed minimum of one labeled node
%                                 per ground-truth community, as used in
%                                 the supervision-ratio sensitivity
%                                 analysis: "labeled nodes were selected
%                                 using stratified random sampling, with
%                                 at least one labeled node selected from
%                                 each ground-truth community."
%
%   [labeledIdx, labeledLabels] = SAMPLE_LABELED_SET_STRATIFIED(allLabels, labelFrac, seed)

if nargin < 3
    seed = 1;
end
rng(seed);

n = numel(allLabels);
classes = unique(allLabels);
target = max(1, round(labelFrac * n));

labeledIdx = [];
for c = classes
    members = find(allLabels == c);
    pick = members(randi(numel(members)));
    labeledIdx = [labeledIdx, pick]; %#ok<AGROW>
end

remaining = setdiff(1:n, labeledIdx);
remaining = remaining(randperm(numel(remaining)));
extra = max(0, target - numel(labeledIdx));
labeledIdx = [labeledIdx, remaining(1:min(extra, numel(remaining)))];
labeledIdx = sort(labeledIdx);
labeledLabels = allLabels(labeledIdx);
end
