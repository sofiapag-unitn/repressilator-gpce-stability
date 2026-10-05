% ========================================================================
% STEADY-STATE SURROGATE: 2D SYMMETRIC vs 6D ASYMMETRIC
% Repressilator | Hill n=3
% ========================================================================
clear; clc; close all;
% Edit the next line to point to your PoCET-master folder
addpath(genpath('C:\path\to\PoCET-master'));
if exist('get_PSImap', 'file') ~= 2
    error('get_PSImap not found: check that the full PoCET toolbox folder (including auxiliary/) is on the path.');
end

disp('================================================================');
disp('STEADY-STATE SURROGATE: 2D SYMMETRIC vs 6D ASYMMETRIC');
disp('================================================================');

%% ── CONFIGURATION ───────────────────────────────────────────────────────
n_hill      = 3;
order       = 3;
N_MC_true   = 5000000; % Number of samples for Reference MC
N_MC_hyb    = 500000; % Number of samples for Surrogate gPCE
chunk_size  = 10000;   % Vectorization chunk size
fzero_opts  = optimset('Display','off');

% Bounds
bnd_alpha = [0.100, 1.950];
bnd_delta = [0.140, 0.700];

bounds_2D = [bnd_alpha; bnd_delta];
bounds_6D = [bnd_alpha; bnd_alpha; bnd_alpha; bnd_delta; bnd_delta; bnd_delta];

%% ── LOCAL HELPERS ───────────────────────────────────────────────────────
% Map standard variables [-1, 1] to physical bounds
function params = xi_to_params(xi_row, bounds)
    n = size(xi_row, 2);
    params = zeros(size(xi_row,1), n);
    for i = 1:n
        a = bounds(i,1); b = bounds(i,2);
        params(:,i) = (a+b)/2 + xi_row(:,i)*(b-a)/2;
    end
end

% 1D Gauss-Legendre Quadrature Nodes
function [x, w] = gauss_legendre_1d(n)
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    T    = diag(beta, 1) + diag(beta, -1);
    [V, D] = eig(T);
    x = diag(D);
    [x, idx] = sort(x);
    w = 2 * V(1, idx)'.^2;
end

% 6D Full Model Evaluation
function [max_ev, x3_star] = eval_system_6D(params, n_hill, fzero_opts)
    a1_sq = params(1); a2_sq = params(2); a3_sq = params(3);
    d1    = params(4); d2    = params(5); d3    = params(6);
    g1 = @(x) a1_sq / (1 + x^n_hill);
    g2 = @(x) a2_sq / (1 + x^n_hill);
    g3 = @(x) a3_sq / (1 + x^n_hill);
    map_x1  = @(x3) g1(x3) / d1;
    map_x2  = @(x3) g2(map_x1(x3)) / d2;
    obj_fun = @(x3) x3 - (g3(map_x2(x3)) / d3);
    
    x3_star = fzero(obj_fun, [0, 1000], fzero_opts);
    x1_star = map_x1(x3_star);
    x2_star = map_x2(x3_star);
    
    k1 = (n_hill * a1_sq * x3_star^(n_hill-1)) / ((1 + x3_star^n_hill)^2);
    k2 = (n_hill * a2_sq * x1_star^(n_hill-1)) / ((1 + x1_star^n_hill)^2);
    k3 = (n_hill * a3_sq * x2_star^(n_hill-1)) / ((1 + x2_star^n_hill)^2);
    
    J = [-d1, 0, -k1; -k2, -d2, 0; 0, -k3, -d3];
    max_ev = max(real(eig(J)));
end

% 2D Symmetric Model Evaluation
function [k, x_star] = eval_system_2D(params, n_hill, fzero_opts)
    a_sq = params(1);
    d    = params(2);
    obj_fun = @(x) x - (a_sq / (d * (1 + x^n_hill)));
    x_star = fzero(obj_fun, [0, 1000], fzero_opts);
    k = (n_hill * a_sq * x_star^(n_hill-1)) / ((1 + x_star^n_hill)^2);
end

%% ========================================================================
%  PART 1: 2D SYMMETRIC SETUP & TRAINING
%  ========================================================================
fprintf('\n--- 2D SYMMETRIC MODEL ---\n');
fprintf('Building 2D Quadrature Grid (900 pts)...\n');
[xi_1d_30, w_1d_30] = gauss_legendre_1d(30); % 30x30 = 900 pts
[gq1_2d, gq2_2d] = ndgrid(xi_1d_30, xi_1d_30);
xi_quad_2D = [gq1_2d(:), gq2_2d(:)];
w_nd_2D = kron(w_1d_30, w_1d_30);
weights_q_2D = w_nd_2D(:) / (2^2);
N_quad_2D = size(xi_quad_2D, 1);

true_x_star_2D = zeros(N_quad_2D, 1);
for i = 1:N_quad_2D
    p_ = xi_to_params(xi_quad_2D(i,:), bounds_2D);
    [~, x_star] = eval_system_2D(p_, n_hill, fzero_opts);
    true_x_star_2D(i) = x_star;
end

sys_2D = struct();
sys_2D.pce.options.n_xi = 2;
sys_2D.pce.options.order = order;
sys_2D.randomVariables = 1:2;
sys_2D.parameters(1).name = 'alpha_sq'; sys_2D.parameters(1).dist = 'uniform';
sys_2D.parameters(2).name = 'delta';    sys_2D.parameters(2).dist = 'uniform';
sys_2D.pce.vars = {'alpha_sq', 'delta'};
sys_2D.states(1).name = 'dummy';
basis_2D = PoCETbasis(sys_2D);
test_args_2D = num2cell(zeros(1, 2));
p_basis_2D = numel(basis_2D(test_args_2D{:}));
basis_func_2D = @(xi) basis_2D(xi(1), xi(2));

A_2D = zeros(N_quad_2D, p_basis_2D);
for i = 1:N_quad_2D
    A_2D(i,:) = basis_func_2D(xi_quad_2D(i,:))';
end
coeffs_x_2D = ( (A_2D .* weights_q_2D)' * A_2D ) \ ( (A_2D .* weights_q_2D)' * true_x_star_2D );

%% ========================================================================
%  PART 2: 6D ASYMMETRIC SETUP & TRAINING
%  ========================================================================
fprintf('--- 6D ASYMMETRIC MODEL ---\n');
fprintf('Building 6D Quadrature Grid (4096 pts)...\n');
[xi_1d_4, w_1d_4] = gauss_legendre_1d(4); % 4^6 = 4096 pts
[gq1,gq2,gq3,gq4,gq5,gq6] = ndgrid(xi_1d_4,xi_1d_4,xi_1d_4,xi_1d_4,xi_1d_4,xi_1d_4);
xi_quad_6D = [gq1(:),gq2(:),gq3(:),gq4(:),gq5(:),gq6(:)]; 
w_nd_6D = w_1d_4;
for i = 2:6; w_nd_6D = kron(w_nd_6D, w_1d_4); end
weights_q_6D = w_nd_6D(:) / (2^6);
N_quad_6D = size(xi_quad_6D, 1);

true_x3_star_6D = zeros(N_quad_6D, 1);
for i = 1:N_quad_6D
    p_ = xi_to_params(xi_quad_6D(i,:), bounds_6D);
    [~, x3] = eval_system_6D(p_, n_hill, fzero_opts);
    true_x3_star_6D(i) = x3;
end

sys_6D = struct();
sys_6D.pce.options.n_xi = 6;
sys_6D.pce.options.order = order;
sys_6D.randomVariables = 1:6;
param_names = {'a1','a2','a3','d1','d2','d3'};
for i = 1:6
    sys_6D.parameters(i).name = param_names{i};
    sys_6D.parameters(i).dist = 'uniform';
end
sys_6D.pce.vars = param_names;
sys_6D.states(1).name = 'dummy';
basis_6D = PoCETbasis(sys_6D);
test_args_6D = num2cell(zeros(1, 6));
p_basis_6D = numel(basis_6D(test_args_6D{:}));
basis_func_6D = @(xi) basis_6D(xi(1),xi(2),xi(3),xi(4),xi(5),xi(6));

A_6D = zeros(N_quad_6D, p_basis_6D);
for i = 1:N_quad_6D
    A_6D(i,:) = basis_func_6D(xi_quad_6D(i,:))';
end
coeffs_x3_6D = ( (A_6D .* weights_q_6D)' * A_6D ) \ ( (A_6D .* weights_q_6D)' * true_x3_star_6D );

%% ========================================================================
%  PART 3A: TRUE REFERENCE MONTE CARLO (5,000,000 Samples)
%  ========================================================================
fprintf('\nEvaluating True Reference MC (%d samples)...\n', N_MC_true);
n_chunks_true = N_MC_true / chunk_size;

mc_stable_true_2D = 0;
mc_stable_true_6D = 0;

for c = 1:n_chunks_true
    % --- SAMPLE GENERATION ---
    xi_chunk_2D = -1 + 2*rand(chunk_size, 2);
    p_chunk_2D  = xi_to_params(xi_chunk_2D, bounds_2D);
    
    xi_chunk_6D = -1 + 2*rand(chunk_size, 6);
    p_chunk_6D  = xi_to_params(xi_chunk_6D, bounds_6D);
    
    % --- TRUE EVALUATION ---
    for i = 1:chunk_size
        % True 2D
        [k_true, ~] = eval_system_2D(p_chunk_2D(i,:), n_hill, fzero_opts);
        if k_true < 2*p_chunk_2D(i,2); mc_stable_true_2D = mc_stable_true_2D + 1; end
        
        % True 6D
        ev = eval_system_6D(p_chunk_6D(i,:), n_hill, fzero_opts);
        if ev < 0; mc_stable_true_6D = mc_stable_true_6D + 1; end
    end
end

%% ========================================================================
%  PART 3B: HYBRID STEADY-STATE gPCE SURROGATE 
%  ========================================================================
fprintf('Evaluating Hybrid gPCE Surrogate (%d samples)...\n', N_MC_hyb);
n_chunks_hyb = N_MC_hyb / chunk_size;

mc_stable_hyb_2D = 0;
mc_stable_hyb_6D = 0;

for c = 1:n_chunks_hyb
    % --- SAMPLE GENERATION ---
    xi_chunk_2D = -1 + 2*rand(chunk_size, 2);
    p_chunk_2D  = xi_to_params(xi_chunk_2D, bounds_2D);
    
    xi_chunk_6D = -1 + 2*rand(chunk_size, 6);
    p_chunk_6D  = xi_to_params(xi_chunk_6D, bounds_6D);
    
    % --- 2D HYBRID EVALUATION ---
    A_chunk_2D = zeros(chunk_size, p_basis_2D);
    for i = 1:chunk_size
        A_chunk_2D(i,:) = basis_func_2D(xi_chunk_2D(i,:))';
    end
    
    pred_x_2D = max(A_chunk_2D * coeffs_x_2D, 0); 
    a_sq = p_chunk_2D(:,1); d = p_chunk_2D(:,2);
    k_hyb_2D = (n_hill .* a_sq .* pred_x_2D.^(n_hill-1)) ./ ((1 + pred_x_2D.^n_hill).^2);
    mc_stable_hyb_2D = mc_stable_hyb_2D + sum(k_hyb_2D < 2.*d);
    
    % --- 6D HYBRID EVALUATION ---
    A_chunk_6D = zeros(chunk_size, p_basis_6D);
    for i = 1:chunk_size
        A_chunk_6D(i,:) = basis_func_6D(xi_chunk_6D(i,:))';
    end
    
    pred_x3_6D = max(A_chunk_6D * coeffs_x3_6D, 0);
    a1_sq = p_chunk_6D(:,1); a2_sq = p_chunk_6D(:,2); a3_sq = p_chunk_6D(:,3);
    d1 = p_chunk_6D(:,4); d2 = p_chunk_6D(:,5); d3 = p_chunk_6D(:,6);
    
    pred_x1 = (a1_sq ./ (1 + pred_x3_6D.^n_hill)) ./ d1;
    pred_x2 = (a2_sq ./ (1 + pred_x1.^n_hill)) ./ d2;
    
    k1 = (n_hill .* a1_sq .* pred_x3_6D.^(n_hill-1)) ./ ((1 + pred_x3_6D.^n_hill).^2);
    k2 = (n_hill .* a2_sq .* pred_x1.^(n_hill-1)) ./ ((1 + pred_x1.^n_hill).^2);
    k3 = (n_hill .* a3_sq .* pred_x2.^(n_hill-1)) ./ ((1 + pred_x2.^n_hill).^2);
    
    for i = 1:chunk_size
        J = [-d1(i), 0, -k1(i); -k2(i), -d2(i), 0; 0, -k3(i), -d3(i)];
        if max(real(eig(J))) < 0; mc_stable_hyb_6D = mc_stable_hyb_6D + 1; end
    end
end

%% ── RESULTS SUMMARY ─────────────────────────────────────────────────────
prob_true_2D = (mc_stable_true_2D / N_MC_true) * 100;
prob_hyb_2D  = (mc_stable_hyb_2D  / N_MC_hyb) * 100;
err_2D       = prob_hyb_2D - prob_true_2D;

prob_true_6D = (mc_stable_true_6D / N_MC_true) * 100;
prob_hyb_6D  = (mc_stable_hyb_6D  / N_MC_hyb) * 100;
err_6D       = prob_hyb_6D - prob_true_6D;

fprintf('\n================================================================\n');
fprintf('FINAL RESULTS\n');
fprintf('================================================================\n');
fprintf('2D SYMMETRIC MODEL (No Error Amplification):\n');
fprintf('  Reference MC (%d samples):              %.4f%%\n', N_MC_true, prob_true_2D);
fprintf('  Hybrid Steady-State gPCE (%d samples):  %.4f%%  (Error: %+.4f%%)\n', N_MC_hyb, prob_hyb_2D, err_2D);
fprintf('----------------------------------------------------------------\n');
fprintf('6D ASYMMETRIC MODEL (With Error Amplification):\n');
fprintf('  Reference MC (%d samples):              %.4f%%\n', N_MC_true, prob_true_6D);
fprintf('  Hybrid Steady-State gPCE (%d samples):  %.4f%%  (Error: %+.4f%%)\n', N_MC_hyb, prob_hyb_6D, err_6D);
fprintf('================================================================\n');
%% ── 1D SLICE VISUALIZATION: TRUE VS SURROGATE ──────────────────────────
disp('Generating 1D slice comparison...');
n_pts = 100; 
xi_1_sweep = linspace(-1, 1, n_pts)'; 
true_evals = zeros(n_pts, 1);
gpce_evals = zeros(n_pts, 1);
alpha1_vals = zeros(n_pts, 1);

% Fix the other parameters at specific values within [-1, 1]
xi_fixed = [0, 0.8, 0.8, -0.8, -0.8, -0.8];

for i = 1:n_pts
    xi_test = xi_fixed;
    xi_test(1) = xi_1_sweep(i); 
    
    % --- PHYSICAL MAPPING ---
    p_ = xi_to_params(xi_test, bounds_6D);
    alpha1_vals(i) = p_(1); % <--- THIS FIXES THE X-AXIS
    
    % 1. True Eigenvalue
    [true_ev, ~] = eval_system_6D(p_, n_hill, fzero_opts);
    true_evals(i) = true_ev;
    
    % 2. gPCE Predicted Concentration -> Hybrid Eigenvalue
    basis_eval = basis_func_6D(xi_test)';
    pred_x3 = max(basis_eval * coeffs_x3_6D, 0);
    
    % Manually calculate J using the predicted x3
    pred_x1 = (p_(1) / (1 + pred_x3^n_hill)) / p_(4);
    pred_x2 = (p_(2) / (1 + pred_x1^n_hill)) / p_(5);
    
    k1 = (n_hill * p_(1) * pred_x3^(n_hill-1)) / ((1 + pred_x3^n_hill)^2);
    k2 = (n_hill * p_(2) * pred_x1^(n_hill-1)) / ((1 + pred_x1^n_hill)^2);
    k3 = (n_hill * p_(3) * pred_x2^(n_hill-1)) / ((1 + pred_x2^n_hill)^2);
    
    J_hyb = [-p_(4), 0, -k1; -k2, -p_(5), 0; 0, -k3, -p_(6)];
    gpce_evals(i) = max(real(eig(J_hyb))); 
end

% ── Plotting ─────────────────────────────────────────────────────────────
figure('Color', 'w', 'Position', [100, 100, 800, 500]);
plot(alpha1_vals, true_evals, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Model'); hold on;
plot(alpha1_vals, gpce_evals, 'r--', 'LineWidth', 2, 'DisplayName', 'Hybrid gPCE');
yline(0, 'b:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('\alpha_1^2 (Physical Scale)', 'FontSize', 12);
ylabel('Max Real Eigenvalue \lambda_{max}', 'FontSize', 12);
title('Stability Map Slice (6D Asymmetric Model)', 'FontSize', 14);
legend('Location', 'best');
grid on;
