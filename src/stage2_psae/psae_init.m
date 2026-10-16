function model = psae_init(inputDim, branchLatentSizes, fusionDim, numClasses, outputActivation)
% PSAE_INIT  Parallel stacked autoencoder with class-associated branch masks.
%
%   Each branch b is associated with class c = mod(b-1, K)+1. Feature masks
%   are initialized complementary; call APPLY_CLASS_MASKS after labeled
%   data is known to specialize masks by class-discriminative features.

B = numel(branchLatentSizes);
model.B = B;
model.inputDim = inputDim;
model.fusionDim = fusionDim;
model.numClasses = numClasses;

model.branches = cell(B, 1);
model.branchInputMasks = cell(B, 1);
model.branchClassId = zeros(1, B);  % class association for each branch
totalLatent = 0;
for b = 1:B
    sizes = [inputDim, branchLatentSizes{b}];
    model.branches{b} = init_branch(sizes, outputActivation);
    totalLatent = totalLatent + sizes(end);
    model.branchClassId(b) = mod(b - 1, numClasses) + 1;
    % Initial complementary mask (~60% keep); refined by apply_class_masks
    mask = (rand(inputDim, 1) < 0.6);
    if ~any(mask), mask(1) = true; end
    model.branchInputMasks{b} = double(mask);
end
% Every feature covered by at least one branch
covered = false(inputDim, 1);
for b = 1:B
    covered = covered | (model.branchInputMasks{b} > 0);
end
missing = find(~covered);
for ii = 1:numel(missing)
    b = mod(ii - 1, B) + 1;
    model.branchInputMasks{b}(missing(ii)) = 1;
end

model.fusion.W = init_he(fusionDim, totalLatent);
model.fusion.b = zeros(fusionDim, 1);
model.fusion.activation = 'relu';

model.clf.W = init_he(numClasses, fusionDim);
model.clf.b = zeros(numClasses, 1);
end
