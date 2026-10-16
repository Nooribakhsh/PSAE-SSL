function idx = alias_draw(J, q)
% ALIAS_DRAW  Draw one index in O(1) using alias tables (J, q) produced
%             by ALIAS_SETUP.

K = numel(J);
kk = floor(rand() * K) + 1;
if rand() < q(kk)
    idx = kk;
else
    idx = J(kk);
end
end
