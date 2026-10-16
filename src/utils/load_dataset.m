function [adjList, labels, K] = load_dataset(name, dataRoot)
% LOAD_DATASET  Load a benchmark dataset from data/<name>/.
%
%   [adjList, labels, K] = LOAD_DATASET(name, dataRoot)
%
%   name     : karate, dolphins (or dolphin), football, polbooks, polblogs,
%              email, cora, citeseer, pubmed or dblp
%   dataRoot : path to the data folder
%
% Directed graphs (Table 2): Polblogs, Email, CiteSeer kept directed.

key = lower(strtrim(name));
if strcmp(key, 'dolphin'), key = 'dolphins'; end

directedSets = {'polblogs', 'email', 'citeseer'};
directed = any(strcmp(key, directedSets));

folder = fullfile(dataRoot, key);
% fall back to singular folder name if present
if ~exist(folder, 'dir') && strcmp(key, 'dolphins')
    alt = fullfile(dataRoot, 'dolphin');
    if exist(alt, 'dir'), folder = alt; end
end

labelFile = fullfile(folder, 'labels.txt');
edgeFile  = fullfile(folder, 'edges.txt');

if exist(labelFile, 'file')
    labels = load_labels(labelFile);
    n = numel(labels);
else
    labels = [];
    fid=fopen(edgeFile,'r'); C=textscan(fid,'%f %f','CommentStyle','#','CollectOutput',true); fclose(fid); E=C{1};
    n = max(E(:));
end

adjList = load_graph(edgeFile, n, directed);

if isempty(labels)
    K = 0;
else
    K = numel(unique(labels));
end
% C1 (DBLP): labels are SNAP community ids (1..13477) via largest-community
% assignment on com-dblp.all.cmty.txt. Overlapping community count is 13477
% (paper Table 2). Exclusive primary labels cover fewer classes (~11381);
% we still set K = 13477 so model output dim matches the paper.
if strcmp(key, 'dblp')
    K = 13477;
end
end
