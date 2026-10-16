function labels = load_labels(labelFile)
% LOAD_LABELS  Read a one-integer-per-line ground-truth community label
%              file (node i's label on line i).
fid = fopen(labelFile, 'r');
if fid < 0
    error('load_labels:openFailed', 'Cannot open label file: %s', labelFile);
end
C = textscan(fid, '%f', 'CommentStyle', '#', 'CollectOutput', true);
fclose(fid);
labels = C{1};
labels = labels(:)';
end
