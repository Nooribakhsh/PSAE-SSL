function s = score_partition(adjList, labels, labeledIdx, labeledLabels, dsName)
% Higher is better: primarily labeled accuracy, then modularity.
labels = labels(:)';
agree = 0;
if ~isempty(labeledIdx)
    for i = 1:numel(labeledIdx)
        if labels(labeledIdx(i)) == labeledLabels(i)
            agree = agree + 1;
        end
    end
    agree = agree / numel(labeledIdx);
else
    agree = 1;
end
if nargin >= 5 && ~isempty(dsName)
    Q = compute_modularity(adjList, labels, dsName);
else
    Q = compute_modularity(adjList, labels);
end
s = agree * 10 + Q;  % label agreement dominates
end
