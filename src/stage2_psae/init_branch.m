function branch = init_branch(sizes, outputActivation)
% INIT_BRANCH  Build one encoder-decoder branch (f_b, g_b) with L hidden
%              layers, per Algorithm 2 ("Initialize encoder/decoder
%              weights W_b and biases b_b; He init for ReLU layers;
%              LeCun init for Sigmoid layers").
%
%   branch = INIT_BRANCH(sizes, outputActivation)
%
%   Inputs
%     sizes            : [d0, d1, ..., dL], d0 = input dim (Node2Vec
%                        embedding dim), dL = this branch's latent dim.
%                        The encoder is d0->d1->...->dL (ReLU throughout,
%                        including the latent layer, per Algorithm 2's
%                        "applying ReLU activation" for the encode step).
%                        The decoder mirrors dL->...->d1->d0.
%     outputActivation : 'sigmoid' (bounded targets) or 'linear'
%                        (real-valued targets, e.g. raw Node2Vec inputs)
%                        for the FINAL decoder layer only; all other
%                        decoder layers use ReLU.
%
%   Output
%     branch : struct with fields L, sizes, outputActivation, enc, dec
%              (enc/dec are L x 1 cell arrays of layer structs with
%              fields W, b, activation)

L = numel(sizes) - 1;
branch.L = L;
branch.sizes = sizes;
branch.outputActivation = outputActivation;

% --- Encoder: He init, ReLU at every layer (incl. latent) ---
branch.enc = cell(L, 1);
for l = 1:L
    fan_in = sizes(l);
    fan_out = sizes(l + 1);
    W = init_he(fan_out, fan_in);
    b = zeros(fan_out, 1);
    branch.enc{l} = struct('W', W, 'b', b, 'activation', 'relu');
end

% --- Decoder: mirrored sizes, ReLU hidden layers, chosen output activation ---
decSizes = fliplr(sizes); % [dL, ..., d1, d0]
branch.dec = cell(L, 1);
for l = 1:L
    fan_in = decSizes(l);
    fan_out = decSizes(l + 1);
    if l < L
        act = 'relu';
        W = init_he(fan_out, fan_in);
    else
        act = outputActivation;
        if strcmp(act, 'sigmoid')
            W = init_lecun(fan_out, fan_in);
        else
            W = init_he(fan_out, fan_in);
        end
    end
    b = zeros(fan_out, 1);
    branch.dec{l} = struct('W', W, 'b', b, 'activation', act);
end
end
