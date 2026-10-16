function labels = post_cut_reassignment(Hf, adjList, labels, epsAssign)
% POST_CUT_REASSIGNMENT  Algorithm 6. Re-score each node against all
% communities; accept a move if affinity improves by more than epsAssign.
% After both passes, reject the whole update if modularity decreased.

if nargin < 4
    epsAssign = 1e-6;
end

n = size(Hf, 2);
labels = labels(:)';
if size(Hf, 2) ~= n
    % Hf must be dF x n
    if size(Hf, 1) == n
        Hf = Hf';
    end
end
n = size(Hf, 2);
labels = labels(:)';
if numel(labels) ~= n
    error('post_cut_reassignment:size', 'Hf columns (%d) != labels (%d)', n, numel(labels));
end

Q0 = compute_modularity(adjList, labels);
labelsIn = labels;

for pass = 1:2
    K = max(labels);
    members = cell(1, K);
    for c = 1:K
        members{c} = find(labels == c);
    end
    newLabels = labels;
    for i = 1:n
        bestScore = -Inf;
        bestC = labels(i);
        currentScore = -Inf;
        for c = 1:K
            if isempty(members{c}), continue; end
            score = node_cluster_affinity(Hf, i, members{c}, epsAssign);
            if c == labels(i)
                currentScore = score;
            end
            if score > bestScore
                bestScore = score;
                bestC = c;
            end
        end
        if bestC ~= labels(i) && (bestScore - currentScore) > epsAssign
            newLabels(i) = bestC;
        end
    end
    labels = newLabels;
end

% Accept only if modularity did not decrease
Q1 = compute_modularity(adjList, labels);
if Q1 + 1e-12 < Q0
    labels = labelsIn;
end
end
