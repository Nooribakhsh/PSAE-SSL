function [Lsep, dHf] = separation_loss_and_grad(Hf, classIdxCell, margin)
% SEPARATION_LOSS_AND_GRAD  Margin separation + intra-class compactness.
%
%   L_sep = L_margin + L_compact
%   L_margin  = (2/(K(K-1))) * sum_{a<b} max(0, m - ||mu_a-mu_b||_2^2)
%   L_compact = (1/K_active) * sum_k mean_{i in C_k} ||h_i - mu_k||_2^2
%
% Compactness encourages intra-community cohesion claimed in the method.

K = numel(classIdxCell);
dF = size(Hf, 1);
N  = size(Hf, 2);

mu = zeros(dF, K);
sizes = zeros(1, K);
active = [];
for k = 1:K
    idx = classIdxCell{k};
    if isempty(idx), continue; end
    sizes(k) = numel(idx);
    mu(:, k) = mean(Hf(:, idx), 2);
    active(end+1) = k; %#ok<AGROW>
end
Ka = numel(active);
if Ka < 2
    Lsep = 0;
    dHf = zeros(size(Hf));
    return
end

% ---- margin term (pairwise centroid separation) ----
coeff = 2 / (K * max(K - 1, 1));
Lmargin = 0;
dMu = zeros(dF, K);
for ii = 1:Ka
    for jj = ii+1:Ka
        a = active(ii); b = active(jj);
        diff = mu(:, a) - mu(:, b);
        dist2 = sum(diff .^ 2);
        if margin - dist2 > 0
            Lmargin = Lmargin + coeff * (margin - dist2);
            % d/d mu_a of max(0,m-||d||^2) = -2 diff; chain later to Hf
            dMu(:, a) = dMu(:, a) - coeff * 2 * diff;
            dMu(:, b) = dMu(:, b) + coeff * 2 * diff;
        end
    end
end

% ---- compactness term ----
Lcompact = 0;
dHf = zeros(size(Hf));
for k = active
    idx = classIdxCell{k};
    H = Hf(:, idx);
    resid = H - mu(:, k);
    Lcompact = Lcompact + mean(sum(resid .^ 2, 1));
    % grad w.r.t. each point: 2/|C_k| * resid; also through mu
    dHf(:, idx) = dHf(:, idx) + 0.1 * (2 / sizes(k)) * resid;
end
Lcompact = Lcompact / max(Ka, 1);

% margin grad through mu -> Hf
for k = active
    idx = classIdxCell{k};
    dHf(:, idx) = dHf(:, idx) + dMu(:, k) / sizes(k);
end

Lsep = Lmargin + 0.1 * Lcompact;  % compactness at lower weight so margin dominates
end
