function [J, q] = alias_setup(probs)
% ALIAS_SETUP  Build alias tables (J, q) for O(1) sampling.

probs = probs(:);
K = numel(probs);
if K == 0
    J = zeros(0, 1);
    q = zeros(0, 1);
    return
end
% Guard against NaN/Inf from degenerate normalizations
probs(~isfinite(probs) | probs < 0) = 0;
s = sum(probs);
if s <= 0
    probs = ones(K, 1) / K;
else
    probs = probs / s;
end

q = K * probs;
J = (1:K)';

smaller = find(q < 1.0);
larger  = find(q >= 1.0);

while ~isempty(smaller) && ~isempty(larger)
    small = smaller(end); smaller(end) = [];
    large = larger(end);  larger(end)  = [];

    J(small) = large;
    q(large) = q(large) + q(small) - 1.0;

    if q(large) < 1.0
        smaller(end+1) = large; %#ok<AGROW>
    else
        larger(end+1) = large;  %#ok<AGROW>
    end
end
% Clamp residual mass; only assign when index sets are non-empty
if ~isempty(smaller)
    q(smaller) = 1;
end
if ~isempty(larger)
    q(larger) = 1;
end
end
