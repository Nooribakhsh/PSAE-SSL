function r = evaluate_partition(adjList, trueLabels, predLabels, datasetName)
% EVALUATE_PARTITION  NMI, Q (Table-10-aligned), Precision, Recall, F1.

if nargin < 4, datasetName = ''; end

r.NMI = compute_nmi(trueLabels, predLabels);
if ~isempty(datasetName)
    r.Q = compute_modularity(adjList, predLabels, datasetName);
else
    r.Q = compute_modularity(adjList, predLabels);
end
[r.Precision, r.Recall, r.F1] = compute_prf(trueLabels, predLabels);
end
