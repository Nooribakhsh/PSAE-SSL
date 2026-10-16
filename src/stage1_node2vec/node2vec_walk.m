function walk = node2vec_walk(G, start_node, walk_length)
% NODE2VEC_WALK  Simulate a single length-l biased random walk starting
%                at start_node, using the precomputed 2nd-order alias
%                tables in G (from BUILD_TRANSITION_PROBS). This is the
%                inner "Simulate a biased random walk" step of
%                Algorithm 1 in the paper.
%
%   walk = NODE2VEC_WALK(G, start_node, walk_length)

walk = zeros(1, walk_length);
walk(1) = start_node;

if walk_length == 1
    return
end

node_alias = G.alias_nodes{start_node};
if isempty(node_alias)
    walk = walk(1); % isolated node: cannot extend the walk
    return
end

idx = alias_draw(node_alias.J, node_alias.q);
walk(2) = node_alias.neighbors(idx);

for i = 3:walk_length
    prev = walk(i - 2);
    curr = walk(i - 1);

    curr_nbrs = G.adjList{curr};
    if isempty(curr_nbrs)
        walk = walk(1:i - 1); % dead end: truncate
        return
    end

    eid = full(G.edgeId(prev, curr));
    if eid > 0
        edge_alias = G.alias_edge_list{eid};
        idx = alias_draw(edge_alias.J, edge_alias.q);
        walk(i) = edge_alias.neighbors(idx);
    else
        % Fallback (should not normally trigger): uniform choice
        idx = randi(numel(curr_nbrs));
        walk(i) = curr_nbrs(idx);
    end
end
end
