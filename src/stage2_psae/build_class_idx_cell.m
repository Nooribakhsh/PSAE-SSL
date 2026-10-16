function classIdxCell = build_class_idx_cell(trainIdx, trainLabels, P_idx, P_label, K)
% BUILD_CLASS_IDX_CELL  Build classIdxCell{k} = (L_train ∪ P) restricted
%                        to class k, the node set whose fused embeddings
%                        define centroid mu_k in Equation eq_sep.
%
%   classIdxCell = BUILD_CLASS_IDX_CELL(trainIdx, trainLabels, P_idx, P_label, K)
%
%   trainIdx, trainLabels : ground-truth labeled training nodes (L_train,
%                            i.e. L minus the validation split) and their
%                            labels
%   P_idx, P_label         : pseudo-labeled nodes and labels, from
%                             GENERATE_PSEUDO_LABELS
%   K                       : number of classes

classIdxCell = cell(1, K);
for k = 1:K
    a = trainIdx(trainLabels == k);
    b = P_idx(P_label == k);
    classIdxCell{k} = union(a, b);
end
end
