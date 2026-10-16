function W = init_he(fan_out, fan_in)
% INIT_HE  He initialization for ReLU layers (Equation eq_he):
%          W ~ N(0, 2/fan_in)
W = randn(fan_out, fan_in) * sqrt(2 / fan_in);
end
