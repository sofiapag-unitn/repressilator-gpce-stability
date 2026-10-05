% ========================================================================
% REPRESSILATOR TIME-CONSTRAINED STABILITY ANALYSIS
% 6D Asymmetric | Hill n=3
% Methods:  MC, Direct Quadrature, gPCE Surrogate MC
%
% All three methods run under the same time budgets: 1, 5, 10, 20, 30 s.
% N is simply whatever each method accumulates within the clock 

% Quadrature exception: it is a deterministic rule with a fixed cost
% (4096 true-model evaluations). It either fits in the budget or it
% doesn't 
%
% gPCE surrogate: the one-time training cost (4096 true-model evals +
% coefficient solve) is reported separately 
% Ground truth: N = 5,000,000 unconstrained MC
% ========================================================================
clear; clc; close all;

addpath(genpath('C:\Users\pagoa\Downloads\PoCET-master\PoCET-master'));
if exist('get_PSImap', 'file') ~= 2
    error('get_PSImap not found: check that the full PoCET toolbox folder (including auxiliary/) is on the path.');
end

disp('================================================');
disp('REPRESSILATOR TIME-CONSTRAINED ANALYSIS');
disp('6D Asymmetric | Hill n=3');
disp('================================================');

%% ── CONFIGURATION ───────────────────────────────────────────────────────
n_hill      = 3;
n_params    = 6;
n_xi        = n_params;
n_dim       = n_params;
param_names = {'alpha1_sq','alpha2_sq','alpha3_sq','delta1','delta2','delta3'};
bounds      = [0.100, 1.950;
               0.100, 1.950;
               0.100, 1.950;
               0.140, 0.700;
               0.140, 0.700;
               0.140, 0.700];

NQ_FIXED = 4;
order    = 3;
time_budgets = [1, 5, 10, 20, 30];  % seconds 
fzero_opts   = optimset('Display','off');

%% ── LOCAL HELPERS ───────────────────────────────────────────────────────

function max_ev = eval_system(params, n_hill, fzero_opts)
    a1_sq = params(1); a2_sq = params(2); a3_sq = params(3);
    d1    = params(4); d2    = params(5); d3    = params(6);
    g1 = @(x) a1_sq / (1 + x^n_hill);
    g2 = @(x) a2_sq / (1 + x^n_hill);
    g3 = @(x) a3_sq / (1 + x^n_hill);
    map_x1  = @(x3) g1(x3) / d1;
    map_x2  = @(x3) g2(map_x1(x3)) / d2;
    obj_fun = @(x3) x3 - (g3(map_x2(x3)) / d3);

    % Solve for steady state 
    x3_star = fzero(obj_fun, [0, 1000], fzero_opts);
    x1_star = map_x1(x3_star);
    x2_star = map_x2(x3_star);
    k1 = (n_hill * a1_sq * x3_star^(n_hill-1)) / ((1 + x3_star^n_hill)^2);
    k2 = (n_hill * a2_sq * x1_star^(n_hill-1)) / ((1 + x1_star^n_hill)^2);
    k3 = (n_hill * a3_sq * x2_star^(n_hill-1)) / ((1 + x2_star^n_hill)^2);
    J      = [-d1,   0, -k1;
              -k2, -d2,   0;
                0, -k3, -d3];
    max_ev = max(real(eig(J)));
end

 % Transform from standardized ξ to physical parameters
function params = xi_to_params(xi_row, bounds)
    n = numel(xi_row);
    params = zeros(1, n);
    for i = 1:n
        a = bounds(i,1); b = bounds(i,2);
        params(i) = (a+b)/2 + xi_row(i)*(b-a)/2;
    end
end
%returns two column vectors: x (the evaluation points, or "nodes")
% w (the integration weights)
function [x, w] = gauss_legendre_1d(n)
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    T    = diag(beta, 1) + diag(beta, -1); % constructs the Jacobi matrix (T)
    [V, D] = eig(T);
    %the eigenvalues of the Jacobi matrix are exactly the roots of the n-th
    %Legendre polynomial
    %eigenvalues are our quadrature nodes
    x = diag(D);
    [x, idx] = sort(x);
    w = 2 * V(1, idx)'.^2;
    % the Golub-Welsch theorem states that the weight wj​ for a specific node xj​ 
    % is equal to the integral of the weight function over the interval
    % multiplied by the squared first element of the j-th normalized eigenvector
end

%% ── GROUND TRUTH: 5M-sample Reference MC (unconstrained) ───────────────
disp('------------------------------------------------');
disp('GROUND TRUTH  (N = 5,000,000, no time limit)');

N_ground  = 5000000;
gt_stable = 0;
tic_gt    = tic;
for i = 1:N_ground
    xi = -1 + 2*rand(1, n_xi);
    p_ = xi_to_params(xi, bounds);
    ev = eval_system(p_, n_hill, fzero_opts);
    if ev < 0; gt_stable = gt_stable + 1; end
end
P_ground_truth = gt_stable / N_ground;
fprintf('P_stable = %.6f%%   (%.1f s)\n', P_ground_truth*100, toc(tic_gt));

%% ── BUILD QUADRATURE GRID (shared by Methods 2 and 3) ──────────────────
% This is pre-computed once 
disp('------------------------------------------------');
fprintf('BUILDING QUADRATURE GRID  nq=%d  (%d pts)\n', NQ_FIXED, NQ_FIXED^n_dim);

[xi_1d_q, w_1d_q] = gauss_legendre_1d(NQ_FIXED);
[gq1,gq2,gq3,gq4,gq5,gq6] = ndgrid(xi_1d_q,xi_1d_q,xi_1d_q,xi_1d_q,xi_1d_q,xi_1d_q);
xi_quad   = [gq1(:),gq2(:),gq3(:),gq4(:),gq5(:),gq6(:)]; %reshape each 6D array to a column, stack side by side
w_nd_q    = w_1d_q;
for i = 2:n_dim; w_nd_q = kron(w_nd_q, w_1d_q); end
weights_q = w_nd_q(:) / (2^n_dim); % divide by 2⁶=64 so weights sum to 1
N_quad    = size(xi_quad, 1);

max_evs_quad = zeros(N_quad, 1);
tic_quad_build = tic;
for k = 1:N_quad
    p_ = xi_to_params(xi_quad(k,:), bounds);
    max_evs_quad(k) = eval_system(p_, n_hill, fzero_opts);
end
time_quad_build = toc(tic_quad_build);   % wall time for 4096 true-model evals

prob_quad = sum(weights_q(max_evs_quad < 0));
fprintf('Quadrature grid evaluation: %.2f s\n', time_quad_build);
fprintf('Quadrature  P_stable = %.6f%%\n', prob_quad*100);

%% ── BUILD gPCE SURROGATE (coefficients only, also pre-computed once) ────
disp('------------------------------------------------');
disp('BUILDING gPCE SURROGATE BASIS + COEFFICIENTS');

sys = struct();
sys.pce.options.n_xi  = n_xi;
sys.pce.options.order = order;
sys.randomVariables   = 1:n_xi;
for i = 1:n_xi
    sys.parameters(i).name = param_names{i};
    sys.parameters(i).dist = 'uniform';
end
sys.pce.vars       = param_names;
sys.states(1).name = 'dummy';

basis_func = PoCETbasis(sys);
test_args  = num2cell(zeros(1, n_xi));
p_basis    = numel(basis_func(test_args{:}));
basis_func_vec = @(xi_vec) basis_func(xi_vec(1),xi_vec(2),xi_vec(3), ...
                                      xi_vec(4),xi_vec(5),xi_vec(6));
fprintf('Basis dimension  p = %d\n', p_basis);

tic_coeff = tic;
% Build design matrix
% evaluate all basis functions at all quadrature points.
A = zeros(N_quad, p_basis);
for i = 1:N_quad
    A(i,:) = basis_func_vec(xi_quad(i,:))';
end
fprintf('Design matrix built  (%d x %d)\n', N_quad, p_basis);

% Weighted least squares: solve (A'WA)c = A'Wy
% Avoid forming diag(W) explicitly 
Aw     = A .* weights_q;          % N x p, each row scaled by its weight
AtWA   = Aw' * A;                 % p x p
AtWy   = Aw' * max_evs_quad;            % p x 1
coeffs = AtWA \ AtWy;
coeffs = coeffs(:);
time_coeff_solve = toc(tic_coeff);

% Sanity check
cond_A = cond(A);
fprintf('Condition number        = %.2e\n', cond_A);
if cond_A > 1e10
    warning('High condition number — surrogate may be unreliable.');
end

% Total one-time surrogate setup cost = grid evals + coefficient solve
time_surrogate_setup = time_quad_build + time_coeff_solve;
fprintf('Surrogate setup total   = %.2f s  (grid: %.2f s + coeff: %.2f s)\n', ...
        time_surrogate_setup, time_quad_build, time_coeff_solve);

%% ── TIME-CONSTRAINED EXPERIMENT ─────────────────────────────────────────
% Method 1 (MC):        
% Method 2 (Quadrature): fixed cost = time_quad_build; 
% Method 3 (gPCE+MC):   
n_budgets = numel(time_budgets);

res_mc_prob   = zeros(1, n_budgets);
res_mc_n      = zeros(1, n_budgets);
res_gp_prob   = zeros(1, n_budgets);
res_gp_n      = zeros(1, n_budgets);

for b = 1:n_budgets
    T_budget = time_budgets(b);
    fprintf('\n================================================\n');
    fprintf('TIME BUDGET: %d s\n', T_budget);
    fprintf('================================================\n');

    % ── Method 1: Reference MC ──────────────────────────────────────
    mc_stable = 0;  mc_count = 0;
    tic_mc = tic;
    while toc(tic_mc) < T_budget
        xi = -1 + 2*rand(1, n_xi);
        p_ = xi_to_params(xi, bounds);
        ev = eval_system(p_, n_hill, fzero_opts);
        if ev < 0; mc_stable = mc_stable + 1; end
        mc_count = mc_count + 1;
    end
    res_mc_prob(b) = mc_stable / mc_count;
    res_mc_n(b)    = mc_count;
    fprintf('[MC]    N = %8d  P_stable = %8.4f%%  Error = %+.4f%%\n', ...
        mc_count, res_mc_prob(b)*100, (res_mc_prob(b)-P_ground_truth)*100);

    % ── Method 3: gPCE Surrogate MC ─────────────────────────────────
    % Surrogate is already built 
    % Build A_chunk for the whole batch at once then multiply by coeffs.
    gp_stable = 0;  gp_count = 0;
    chunk_gp  = 1000;   % small chunk so toc() is checked frequently
    tic_gp = tic;
    while toc(tic_gp) < T_budget
        xi_s    = -1 + 2*rand(chunk_gp, n_xi);
        A_chunk = zeros(chunk_gp, p_basis);
        for i = 1:chunk_gp
            A_chunk(i,:) = basis_func_vec(xi_s(i,:))';
        end
        sv        = A_chunk * coeffs;
        gp_stable = gp_stable + sum(sv < 0);
        gp_count  = gp_count  + chunk_gp;
    end
    res_gp_prob(b) = gp_stable / gp_count;
    res_gp_n(b)    = gp_count;
    fprintf('[gPCE]  N = %8d  P_stable = %8.4f%%  Error = %+.4f%%\n', ...
        gp_count, res_gp_prob(b)*100, (res_gp_prob(b)-P_ground_truth)*100);
end

%% ── SUMMARY TABLE ───────────────────────────────────────────────────────
W = 115;
fprintf('\n%s\n', repmat('=', 1, W));
fprintf('SUMMARY\n');
fprintf('Ground Truth   P_stable = %.6f%%  (N = %d, unconstrained)\n', ...
        P_ground_truth*100, N_ground);
fprintf('Quadrature     P_stable = %.6f%%  (%d pts, wall time = %.2f s)\n', ...
        prob_quad*100, N_quad, time_quad_build);
fprintf('%s\n', repmat('=', 1, W));
fprintf('%-7s | %-44s | %-24s | %-36s\n', ...
    'Budget', 'Method 1: Reference MC', ...
    'Method 2: Quadrature', 'Method 3: gPCE Surrogate MC');
fprintf('%-7s | %-44s | %-24s | %-36s\n', '(s)', ...
    'P_stable(%)    Error(%)          N', ...
    'P_stable(%)    Error(%)', ...
    'P_stable(%)    Error(%)          N');
fprintf('%s\n', repmat('-', 1, W));

for b = 1:n_budgets
    T    = time_budgets(b);
    mc_e = (res_mc_prob(b) - P_ground_truth)*100;
    qd_e = (prob_quad      - P_ground_truth)*100;
    gp_e = (res_gp_prob(b) - P_ground_truth)*100;

    fprintf('%-7d | %10.4f%%  %+8.4f%%  %10d | %10.4f%%  %+8.4f%% | %10.4f%%  %+8.4f%%  %10d\n', ...
        T, res_mc_prob(b)*100, mc_e, res_mc_n(b), ...
        prob_quad*100, qd_e, ...
        res_gp_prob(b)*100, gp_e, res_gp_n(b));
end
fprintf('%s\n', repmat('=', 1, W));

%% ── 1D SLICE VISUALIZATION: TRUE VS SURROGATE ──────────────────────────
disp('Generating 1D slice comparison...');

n_pts = 500;
xi_1_sweep = linspace(-1, 1, n_pts)'; % Sweep xi from lower to upper bound (-1 to 1)

true_evals = zeros(n_pts, 1);
gpce_evals = zeros(n_pts, 1);
alpha1_vals = zeros(n_pts, 1);

% Fix the other 5 parameters 
% Set alpha2 and alpha3 to be high (+0.8 in standard space)
% Set delta1, delta2, delta3 to be low (-0.8 in standard space)
xi_fixed = [0, 0.8, 0.8, -0.8, -0.8, -0.8];

for i = 1:n_pts
    % 1. Create the 6D point with 5 parameters fixed 
    xi_test = xi_fixed;
    xi_test(1) = xi_1_sweep(i); % Sweep only the first parameter (alpha1_sq)
    
    % 2. Convert standard xi to physical parameters
    p_ = xi_to_params(xi_test, bounds);
    alpha1_vals(i) = p_(1); % Store the physical value for the x-axis
    
    % 3. Evaluate True System
    true_evals(i) = eval_system(p_, n_hill, fzero_opts);
    
    % 4. Evaluate gPCE Surrogate
    basis_eval = basis_func_vec(xi_test)';
    gpce_evals(i) = basis_eval * coeffs;
end

% ── Plotting ─────────────────────────────────────────────────────────────
figure('Color', 'w', 'Position', [100, 100, 700, 500]);

plot(alpha1_vals, true_evals, 'k-', 'LineWidth', 2); hold on;
plot(alpha1_vals, gpce_evals, 'r--', 'LineWidth', 2);

% Add stability boundary line (y = 0)
yline(0, 'b:', 'LineWidth', 1.5, 'Label', 'Stability Boundary', ...
    'LabelHorizontalAlignment', 'left');

% Formatting
xlabel('Physical Parameter: \alpha_1^2', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Maximum Real Eigenvalue', 'FontSize', 12, 'FontWeight', 'bold');
title('1D Slice: True Model vs 3rd-Order gPCE Surrogate', 'FontSize', 14);
legend('True Model', 'gPCE Surrogate', 'Location', 'Best', 'FontSize', 11);
grid on;
box on;

% Constrain x-axis to the physical bounds of alpha1_sq
xlim([bounds(1,1), bounds(1,2)]);