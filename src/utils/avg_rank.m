function r = avg_rank(x)
% AVG_RANK  Ranks of vector x (ascending), ties get the average rank.
%           No toolbox needed.
x = x(:);
n = numel(x);
[xs, ord] = sort(x);
rs = zeros(n, 1);
i = 1;
while i <= n
    j = i;
    while j < n && xs(j + 1) == xs(i), j = j + 1; end
    rs(i:j) = (i + j) / 2;
    i = j + 1;
end
r = zeros(n, 1);
r(ord) = rs;
end
