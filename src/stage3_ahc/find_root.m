function [r, parent] = find_root(parent, x)
% FIND_ROOT  Union-find find with path compression (A4).
% Returns updated parent for the caller to store.
r = x;
while parent(r) ~= r
    r = parent(r);
end
% Path compression
while parent(x) ~= x
    nxt = parent(x);
    parent(x) = r;
    x = nxt;
end
end
