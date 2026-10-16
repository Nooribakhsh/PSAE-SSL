function n2v = get_n2v_options(cfg, n, seed)
% GET_N2V_OPTIONS  Node2Vec options of a dataset with the central epoch budget
%   (get_training_budget.m / get_user_settings.m) applied, so that every
%   experiment script produces the same embedding as psae_pipeline.
n2v = cfg.n2v;
bud = get_training_budget(cfg.name, n);
n2v.numEpochs = bud.skipgramEpochs;
n2v.seed = seed;
end
