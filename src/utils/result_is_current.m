function tf = result_is_current(file, numSeeds)
% RESULT_IS_CURRENT  true if <file> exists and was produced with numSeeds seeds.
%   Used to skip finished experiments but recompute them when the number of
%   seeds in get_user_settings.m was changed.
tf = false;
if ~exist(file, 'file'), return; end
try
    w = whos('-file', file, 'numSeeds');
    if isempty(w), tf = true; return; end   % older file without the field
    S = load(file, 'numSeeds');
    tf = (S.numSeeds == numSeeds);
catch
    tf = false;
end
if ~tf
    fprintf('  (%s was made with a different number of seeds -> recomputing)\n', file);
end
end
