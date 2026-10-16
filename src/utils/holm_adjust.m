function padj = holm_adjust(p)
% HOLM_ADJUST  Holm step-down adjusted p-values (NaN entries ignored).
padj = nan(size(p));
ok = find(~isnan(p));
m = numel(ok);
if m == 0, return; end
[ps, o] = sort(p(ok));
adj = zeros(m, 1);
runmax = 0;
for i = 1:m
    runmax = max(runmax, (m - i + 1) * ps(i));
    adj(i) = min(1, runmax);
end
tmp = zeros(m, 1); tmp(o) = adj;
padj(ok) = tmp;
end
