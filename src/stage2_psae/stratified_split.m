function [trainIdx, valIdx] = stratified_split(labeledIdx, labels, valFrac, seed)
% STRATIFIED_SPLIT  Split labeled nodes into train/val with per-class
%                   stratification. Handles classes with 0 or 1 members.

if nargin < 4
    seed = 1;
end
rng(seed);

labeledIdx = labeledIdx(:)';
labels = labels(:)';
classes = unique(labels);
trainIdx = [];
valIdx = [];
for c = classes(:)'
    members = labeledIdx(labels == c);
    members = members(:)';
    if isempty(members)
        continue
    end
    members = members(randperm(numel(members)));
    if numel(members) == 1
        % Keep the single example in train; val stays empty for this class
        trainIdx = [trainIdx, members]; %#ok<AGROW>
        continue
    end
    nval = max(1, round(valFrac * numel(members)));
    nval = min(nval, numel(members) - 1); % keep >=1 train
    nval = max(nval, 0);
    valIdx = [valIdx, members(1:nval)]; %#ok<AGROW>
    trainIdx = [trainIdx, members(nval + 1:end)]; %#ok<AGROW>
end
% Fallback: if everything landed in val (degenerate), move one back
if isempty(trainIdx) && ~isempty(valIdx)
    trainIdx = valIdx(1);
    valIdx = valIdx(2:end);
end
end
