function usr = get_user_settings()
% GET_USER_SETTINGS  The ONE place that holds every run setting.
%
%   Edit the values below, save, and run any script in experiments/.
%   Every script and every fresh MATLAB worker process reads this file.
%   Every value is written out explicitly; nothing is hidden elsewhere.
%
%   Results go to  results/ . If you later change a setting that influences
%   the results, the new results go to  results_cfgXXXXXX/  automatically
%   (see get_results_dir.m), so runs made with different settings never mix.

%% ---------------- Datasets & download ----------------
usr.datasets = {'karate','dolphins','football','polbooks','polblogs', ...
                'email','cora','citeseer','pubmed','dblp'};
% usr.datasets = {'karate','dolphins','football'};     % quick test

% Download missing datasets automatically from Zenodo before running.
usr.autoDownload  = false;
usr.forceDownload = false;          % true => download again even if data/ exists
usr.zenodoRecord  = '18775800';     % https://doi.org/10.5281/zenodo.18775800

%% ---------------- Protocol ----------------
usr.numSeeds  = 10;     % independent runs per experiment (seeds 1..numSeeds)
usr.mainRuns  = 10;     % runs that enter the main tables
usr.statSeeds = 30;     % runs per dataset in statistical_tests
usr.labelFrac = 0.10;   % fraction of labeled nodes

%% ---------------- Safety gate (must set to 'yes' after review) ----------------
% Set to 'yes' only after you have reviewed every value in this file.
% Experiment scripts refuse to run while this remains 'no'.
usr.settingsConfirmed = 'yes';

usr.statDatasets = {'karate','dolphins','football','polbooks','polblogs', ...
                    'email','cora','citeseer','pubmed','dblp'};

%% ---------------- Training ----------------
usr.patience = 10;      % early stopping patience (epochs)
usr.lr       = 1e-3;    % Adam learning rate
usr.valFrac  = 0.15;    % validation share of the labeled nodes

usr.beta1    = 0.9;     % Adam beta1
usr.beta2    = 0.999;   % Adam beta2
usr.epsAdam  = 1e-8;    % Adam epsilon

usr.ftLrScale        = 0.1;   % fine-tuning learning rate = lr * ftLrScale
usr.sepWarmupEpochs  = 20;    % linear warm-up of the separation loss (epochs)
usr.pseudoT          = 5;     % pseudo-label refresh interval (epochs)
usr.pseudoRetain     = 0.20;  % fraction of nodes kept per class as pseudo-labels

%% ---------------- Epoch budgets (by graph size n) ----------------
% The first row whose maxN >= n is used.
%   skipgram : Node2Vec Skip-gram epochs
%   maxPre   : maximum pre-training epochs
%   maxFt    : maximum fine-tuning epochs
%   minFt    : minimum fine-tuning epochs
usr.epochs = struct( ...
    'maxN',     {200,  2000, 10000, Inf}, ...
    'skipgram', {5,    20,   20,    20 }, ...
    'maxPre',   {120,  400,  1000,  1000}, ...
    'maxFt',    {120,  400,  1000,  1000}, ...
    'minFt',    {20,   40,   80,    80 });

%% ---------------- Model hyper-parameters ----------------
usr.B         = 3;      % number of parallel branches
usr.lambdaRec = 1;      % lambda_rec
usr.lambdaSup = 0.5;    % lambda_sup
usr.lambdaSep = 1;      % lambda_sep
usr.margin    = 1;      % separation margin
usr.wd        = 1e-4;   % weight decay
usr.n2vP      = 1;      % Node2Vec return parameter p
usr.n2vQ      = 1;      % Node2Vec in-out parameter q
usr.n2vNumNegative = 5;       % Skip-gram negative samples
usr.n2vLearningRate = 0.025;  % Skip-gram learning rate

%% ---------------- Per-dataset architecture ----------------
% layers    : encoder hidden widths (input width = d)
% batchSize : 0 = full batch
% d,l,r,w   : Node2Vec dimension, walk length, walks per node, window size
usr.dataset.karate   = struct('layers',[32 16],       'batchSize',0,   'd',64,  'l',40, 'r',10, 'w',5);
usr.dataset.dolphins = struct('layers',[32 16],       'batchSize',0,   'd',64,  'l',40, 'r',10, 'w',5);
usr.dataset.football = struct('layers',[64 32],       'batchSize',0,   'd',128, 'l',40, 'r',10, 'w',5);
usr.dataset.polbooks = struct('layers',[64 32],       'batchSize',0,   'd',128, 'l',40, 'r',10, 'w',5);
usr.dataset.polblogs = struct('layers',[64 32 16],    'batchSize',128, 'd',128, 'l',80, 'r',10, 'w',10);
usr.dataset.email    = struct('layers',[64 32 16],    'batchSize',128, 'd',128, 'l',80, 'r',10, 'w',10);
usr.dataset.citeseer = struct('layers',[128 64 32],   'batchSize',128, 'd',128, 'l',80, 'r',10, 'w',10);
usr.dataset.cora     = struct('layers',[128 64 32],   'batchSize',128, 'd',128, 'l',80, 'r',10, 'w',10);
usr.dataset.pubmed   = struct('layers',[2056 512 128],'batchSize',256, 'd',256, 'l',40, 'r',10, 'w',5);
usr.dataset.dblp     = struct('layers',[2056 512 128],'batchSize',256, 'd',256, 'l',40, 'r',5,  'w',5);

%% ---------------- Sensitivity experiments ----------------
usr.branchValues       = 1:5;
usr.labelRatios        = [0.05 0.10 0.20 0.30 0.50];
usr.labelSensDatasets  = {'karate','citeseer','pubmed'};
end
