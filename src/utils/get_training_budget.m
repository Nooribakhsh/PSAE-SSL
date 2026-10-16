function b = get_training_budget(name, n) %#ok<INUSL>
% GET_TRAINING_BUDGET  Epoch budget for a graph with n nodes.
%
%   b = GET_TRAINING_BUDGET(datasetName, n)
%
% The values live in usr.epochs (src/utils/get_user_settings.m); the first
% row whose maxN >= n is used.

usr = get_user_settings();
E = usr.epochs;
k = find([E.maxN] >= n, 1, 'first');
if isempty(k), k = numel(E); end

b = struct('skipgramEpochs', E(k).skipgram, ...
           'maxEpochsPretrain', E(k).maxPre, ...
           'maxEpochsFinetune', E(k).maxFt, ...
           'minEpochsFinetune', E(k).minFt);
if b.minEpochsFinetune > b.maxEpochsFinetune
    b.minEpochsFinetune = b.maxEpochsFinetune;
end
end
