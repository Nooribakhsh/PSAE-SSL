function res = friedman_test(X, higherIsBetter)
% FRIEDMAN_TEST  Friedman test + Iman-Davenport correction.
%   X : N x k matrix (rows = blocks, e.g. datasets; columns = methods).
%   Rows containing NaN are removed (complete cases).
%   res: N, k, avgRanks (1 = best), chi2, df, p, FF, pFF, usedRows
%   Base MATLAB only (gammainc / betainc).
if nargin < 2, higherIsBetter = true; end
used = all(~isnan(X), 2);
X = X(used, :);
[N, k] = size(X);
res.usedRows = find(used); res.N = N; res.k = k;
R = zeros(N, k);
for i = 1:N
    if higherIsBetter, R(i, :) = (k + 1) - avg_rank(X(i, :)');   % best -> rank 1
    else,              R(i, :) = avg_rank(X(i, :)');
    end
end
Rj = mean(R, 1);
res.avgRanks = Rj;
% tie-corrected chi-square
tieSum = 0;
for i = 1:N
    [~, ~, ic] = unique(X(i, :));
    t = accumarray(ic(:), 1);
    tieSum = tieSum + sum(t .^ 3 - t);
end
num = 12 * N * sum((Rj - (k + 1) / 2) .^ 2);
den = k * (k + 1) - tieSum / (N * (k - 1));
res.chi2 = num / den;
res.df = k - 1;
res.p = 1 - gammainc(res.chi2 / 2, res.df / 2);
den2 = N * (k - 1) - res.chi2;
if den2 > 0
    res.FF = (N - 1) * res.chi2 / den2;
    d1 = k - 1; d2 = (k - 1) * (N - 1);
    res.pFF = 1 - betainc(d1 * res.FF / (d1 * res.FF + d2), d1 / 2, d2 / 2);
else
    res.FF = Inf; res.pFF = 0;
end
end
