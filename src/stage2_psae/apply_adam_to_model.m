function [model, moments] = apply_adam_to_model(model, grads, moments, t, hp)
% APPLY_ADAM_TO_MODEL  Apply ADAM_STEP to every trainable parameter of
%                       the PSAE model (all branches, fusion, classifier).
%
%   hp : struct with fields lr, beta1, beta2, eps, weightDecay

B = model.B;
for b = 1:B
    L = model.branches{b}.L;
    for l = 1:L
        m = moments.branches{b}.enc{l};
        g = grads.branches{b}.enc{l};
        [W, bb, mW, vW, mb, vb] = adam_step(model.branches{b}.enc{l}.W, model.branches{b}.enc{l}.b, ...
            g.dW, g.db, m.mW, m.vW, m.mb, m.vb, t, hp.lr, hp.beta1, hp.beta2, hp.eps, hp.weightDecay);
        model.branches{b}.enc{l}.W = W;
        model.branches{b}.enc{l}.b = bb;
        moments.branches{b}.enc{l} = struct('mW', mW, 'vW', vW, 'mb', mb, 'vb', vb);

        m = moments.branches{b}.dec{l};
        g = grads.branches{b}.dec{l};
        [W, bb, mW, vW, mb, vb] = adam_step(model.branches{b}.dec{l}.W, model.branches{b}.dec{l}.b, ...
            g.dW, g.db, m.mW, m.vW, m.mb, m.vb, t, hp.lr, hp.beta1, hp.beta2, hp.eps, hp.weightDecay);
        model.branches{b}.dec{l}.W = W;
        model.branches{b}.dec{l}.b = bb;
        moments.branches{b}.dec{l} = struct('mW', mW, 'vW', vW, 'mb', mb, 'vb', vb);
    end
end

m = moments.fusion; g = grads.fusion;
[W, bb, mW, vW, mb, vb] = adam_step(model.fusion.W, model.fusion.b, g.dW, g.db, ...
    m.mW, m.vW, m.mb, m.vb, t, hp.lr, hp.beta1, hp.beta2, hp.eps, hp.weightDecay);
model.fusion.W = W; model.fusion.b = bb;
moments.fusion = struct('mW', mW, 'vW', vW, 'mb', mb, 'vb', vb);

m = moments.clf; g = grads.clf;
[W, bb, mW, vW, mb, vb] = adam_step(model.clf.W, model.clf.b, g.dW, g.db, ...
    m.mW, m.vW, m.mb, m.vb, t, hp.lr, hp.beta1, hp.beta2, hp.eps, hp.weightDecay);
model.clf.W = W; model.clf.b = bb;
moments.clf = struct('mW', mW, 'vW', vW, 'mb', mb, 'vb', vb);
end
