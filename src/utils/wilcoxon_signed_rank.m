function res = wilcoxon_signed_rank(a, b)
% WILCOXON_SIGNED_RANK  Two-sided paired Wilcoxon signed-rank test.
%   res = WILCOXON_SIGNED_RANK(a, b)  tests H0: median(a-b) = 0.
%   Zero differences are dropped. Exact p-value by dynamic programming over
%   the (doubled) rank sums when the number of non-zero pairs <= 50 (handles
%   ties via average ranks), otherwise normal approximation with tie and
%   continuity correction. Base MATLAB only.
%   res: n, Wplus, Wminus, stat (= min(W+,W-)), p, method, wins, ties,
%        losses, meanDiff
a = a(:); b = b(:);
ok = ~isnan(a) & ~isnan(b);
a = a(ok); b = b(ok);
d = a - b;
res.wins = sum(d > 0); res.ties = sum(d == 0); res.losses = sum(d < 0);
res.meanDiff = mean(d);
d = d(d ~= 0);
n = numel(d);
res.n = n;
if n == 0
    res.Wplus = 0; res.Wminus = 0; res.stat = 0; res.p = 1; res.method = 'none';
    return
end
r = avg_rank(abs(d));
Wp = sum(r(d > 0)); Wm = sum(r(d < 0));
res.Wplus = Wp; res.Wminus = Wm; res.stat = min(Wp, Wm);
total = sum(r); mu = total / 2;
if n <= 50
    r2 = round(2 * r);                   % integer doubled ranks
    T2 = sum(r2);
    cnt = zeros(1, T2 + 1); cnt(1) = 1;  % cnt(s+1) = #sign patterns with sum s
    for i = 1:n
        sh = r2(i);
        cnt(sh + 1:end) = cnt(sh + 1:end) + cnt(1:end - sh);
    end
    sums = 0:T2;
    dev = abs(sums - T2 / 2);
    obs = abs(round(2 * Wp) - T2 / 2);
    res.p = sum(cnt(dev >= obs - 1e-9)) / 2^n;
    res.method = 'exact';
else
    [~, ~, ic] = unique(abs(d));
    t = accumarray(ic, 1);
    v = n * (n + 1) * (2 * n + 1) / 24 - sum(t .^ 3 - t) / 48;
    z = (abs(Wp - mu) - 0.5) / sqrt(v);
    res.p = erfc(max(z, 0) / sqrt(2));
    res.method = 'normal';
end
res.p = min(1, res.p);
end
