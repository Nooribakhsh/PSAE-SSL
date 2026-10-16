function moments = init_adam_moments(model)
% INIT_ADAM_MOMENTS  Build zero-valued Adam moment buffers (m, v) with
%                    the same shape as every trainable parameter in
%                    `model` (all branches' encoder/decoder layers, the
%                    fusion layer, and the classifier).

B = model.B;
moments.branches = cell(B, 1);
for b = 1:B
    L = model.branches{b}.L;
    encM = cell(L, 1);
    decM = cell(L, 1);
    for l = 1:L
        Wenc = model.branches{b}.enc{l}.W;
        benc = model.branches{b}.enc{l}.b;
        encM{l} = struct('mW', zeros(size(Wenc)), 'vW', zeros(size(Wenc)), ...
                          'mb', zeros(size(benc)), 'vb', zeros(size(benc)));

        Wdec = model.branches{b}.dec{l}.W;
        bdec = model.branches{b}.dec{l}.b;
        decM{l} = struct('mW', zeros(size(Wdec)), 'vW', zeros(size(Wdec)), ...
                          'mb', zeros(size(bdec)), 'vb', zeros(size(bdec)));
    end
    moments.branches{b}.enc = encM;
    moments.branches{b}.dec = decM;
end

moments.fusion = struct('mW', zeros(size(model.fusion.W)), 'vW', zeros(size(model.fusion.W)), ...
                         'mb', zeros(size(model.fusion.b)), 'vb', zeros(size(model.fusion.b)));

moments.clf = struct('mW', zeros(size(model.clf.W)), 'vW', zeros(size(model.clf.W)), ...
                      'mb', zeros(size(model.clf.b)), 'vb', zeros(size(model.clf.b)));
end
