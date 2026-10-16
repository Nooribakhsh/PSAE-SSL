function W = init_lecun(fan_out, fan_in)
% INIT_LECUN  LeCun initialization for Sigmoid layers (Equation eq_lecun):
%             W ~ N(0, 1/fan_in)
W = randn(fan_out, fan_in) * sqrt(1 / fan_in);
end
