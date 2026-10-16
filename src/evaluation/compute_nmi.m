function nmi = compute_nmi(trueLabels, predLabels)
% COMPUTE_NMI  Normalized Mutual Information:
%   NMI(Y, C) = 2 * I(Y; C) / (H(Y) + H(C))

trueLabels = trueLabels(:);
predLabels = predLabels(:);
n = numel(trueLabels);

[~, ~, iy] = unique(trueLabels);
[~, ~, ic] = unique(predLabels);
Ky = max(iy);
Kc = max(ic);

contingency = accumarray([iy, ic], 1, [Ky, Kc]);
Pxy = contingency / n;
Px = sum(Pxy, 2);
Pc = sum(Pxy, 1);

I = 0;
for a = 1:Ky
    for b = 1:Kc
        if Pxy(a, b) > 0
            I = I + Pxy(a, b) * log(Pxy(a, b) / (Px(a) * Pc(b) + eps));
        end
    end
end

Hx = -sum(Px(Px > 0) .* log(Px(Px > 0)));
Hc = -sum(Pc(Pc > 0) .* log(Pc(Pc > 0)));

if (Hx + Hc) == 0
    nmi = 1;
else
    nmi = 2 * I / (Hx + Hc);
end
end
