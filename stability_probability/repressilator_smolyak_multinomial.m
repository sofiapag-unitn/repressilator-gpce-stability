% ------------------------------------------------------------------------
% STABILITY ANALYSIS: Smolyak Grid + Exact Multinomial Moments
% (ASYMMETRIC 6D DEPENDENT JACOBIAN FIX, n=3)
% ------------------------------------------------------------------------
clear; clc; close all;

%% 1. SETUP
disp('================================================');
disp('INHIBITORY RING: STABILITY ANALYSIS (SMOLYAK)');
disp('================================================');

% Edit the next line to point to your PoCET-master folder
addpath(genpath('C:\path\to\PoCET-master'));
if exist('get_PSImap', 'file') ~= 2
    error('get_PSImap not found: check that the full PoCET toolbox folder (including auxiliary/) is on the path.');
end
addpath(fullfile(fileparts(mfilename('fullpath')), 'helpers'));

% Set the Hill coefficient to 3 for oscillatory potential
n_hill = 3;  
fprintf('Hill Coefficient (n): %d\n', n_hill);

param_names = {'alpha1_sq','alpha2_sq','alpha3_sq','delta1','delta2','delta3'};
n_params = 6;

% Parameter ranges
bounds = [0.100, 1.950; 0.100, 1.950; 0.100, 1.950; 
          0.140, 0.700; 0.140, 0.700; 0.140, 0.700];

%% 2. GENERATE SMOLYAK GRID 
% This grid is ONLY used to solve for the coefficients (System Solution)
disp('------------------------------------------------');
disp('1. GENERATING SMOLYAK GRID');

smolyak_level = 3; 
[nodes, weights] = smolyak_grid(n_params, smolyak_level);
N_grid = size(nodes, 1);
fprintf('  Generated %d grid points (Level %d)\n', N_grid, smolyak_level);

%% 3. SYSTEM & BASIS CONFIGURATION
order = 3; 
sys = struct();
sys.pce.options.order = order;
sys.pce.options.n_xi = n_params;
sys.pce.vars = param_names;
sys.randomVariables = 1:n_params;

for i = 1:n_params
    sys.parameters(i).name = param_names{i};
    sys.parameters(i).dist = 'uniform'; 
    sys.parameters(i).data = [-1, 1]; 
end
sys.states(1).name = 'dummy'; 

% Generate Basis
sys.basis = PoCETbasis(sys); 
val = sys.basis(0,0,0,0,0,0); 
p = numel(val);
fprintf('  Basis Size p: %d\n', p);

%% 4. EVALUATE & SOLVE COEFFICIENTS 
disp('------------------------------------------------');
disp('2. SOLVING COEFFICIENTS');

% 1. Evaluate Model
max_eigs = zeros(N_grid, 1);
fzero_opts = optimset('Display','off'); % hides fzero output

for k = 1:N_grid
    xi = nodes(k, :);
    params = zeros(1, n_params);
    
    % Transform from standardized to physical parameter bounds
    for i = 1:n_params
        a = bounds(i, 1); b = bounds(i, 2);
        params(i) = (a + b)/2 + xi(i)*(b - a)/2;
    end
    
    a1_sq = params(1); a2_sq = params(2); a3_sq = params(3);
    d1 = params(4); d2 = params(5); d3 = params(6);
    
    % Define the Hill functions dynamically
    g1 = @(x) a1_sq / (1 + x^n_hill);
    g2 = @(x) a2_sq / (1 + x^n_hill);
    g3 = @(x) a3_sq / (1 + x^n_hill);
    
    % Composite mapping logic for fzero
    map_x1 = @(x3) g1(x3) / d1;
    map_x2 = @(x3) g2(map_x1(x3)) / d2;
    obj_fun = @(x3) x3 - (g3(map_x2(x3)) / d3); 
    
    % Numerically solve for the unique steady state 
    x3_star = fzero(obj_fun, [0, 1000], fzero_opts); 
    x1_star = map_x1(x3_star);
    x2_star = map_x2(x3_star);
    
    % Compute the DEPENDENT inhibitory strengths (k_i)
    k1 = (n_hill * a1_sq * x3_star^(n_hill-1)) / ((1 + x3_star^n_hill)^2);
    k2 = (n_hill * a2_sq * x1_star^(n_hill-1)) / ((1 + x1_star^n_hill)^2);
    k3 = (n_hill * a3_sq * x2_star^(n_hill-1)) / ((1 + x2_star^n_hill)^2);
    
    % Build the final Jacobian matrix
    J = [-d1,  0, -k1; 
         -k2, -d2,  0; 
          0, -k3, -d3];
          
    max_eigs(k) = max(real(eig(J)));
end

% 2. Build Basis Matrix A
A = zeros(N_grid, p);
for i = 1:N_grid
    val = sys.basis(nodes(i,1), nodes(i,2), nodes(i,3), nodes(i,4), nodes(i,5), nodes(i,6));
    A(i, :) = val(:)';
end

% 3. Weighted Least Squares (using pinv)
W = diag(weights); 
coeffs = pinv(A' * W * A) * (A' * W * max_eigs);
fprintf('  Mean (Coeff 1): %.6f\n', coeffs(1));

%% 5. MOMENTS
disp('------------------------------------------------');
disp('3. CALCULATING MOMENTS');
filename = 'InhibitoryRingMoments.mat';
max_moment_order = 4;

% --- OFFLINE PHASE  ---
if exist(filename, 'file')
    fprintf('  Loading precomputed matrices from %s...\n', filename);
    load(filename, 'MomMats');
else
    fprintf('  No cache found. Precomputing (this may take a moment)...\n');
    % This handles the "Offline" work 
    MomMats = precompute_MomMats(n_params, order, max_moment_order);
    save(filename, 'MomMats', '-v7.3');
    fprintf('  Calculations cached to %s.\n', filename);
end

% --- ONLINE (Run whenever coeffs change) ---
tic;
m1 = multinomial_moment(coeffs, 1, MomMats);
m2 = multinomial_moment(coeffs, 2, MomMats);
m3 = multinomial_moment(coeffs, 3, MomMats);
m4 = multinomial_moment(coeffs, 4, MomMats);

 % Convert to Central Moments
mean_val = m1;
var_val = max(0, m2 - m1^2);
std_dev = sqrt(var_val);

if std_dev > 1e-12
    skew_val = (m3 - 3*m1*m2 + 2*m1^3) / std_dev^3;
    kurt_val = (m4 - 4*m1*m3 + 6*m1^2*m2 - 3*m1^4) / std_dev^4;
else
    skew_val = 0; kurt_val = 3;
end

fprintf(' Mean: %.6f\n', mean_val);
fprintf(' Var: %.6f\n', var_val);
fprintf(' Skew: %.6f\n', skew_val);
fprintf(' Kurt: %.6f\n', kurt_val); 

%% 6. EDGEWORTH EXPANSION
disp('------------------------------------------------');
disp('4. EDGEWORTH APPROXIMATION');

z = -mean_val / std_dev;

% Standard normal functions
phi = @(x) exp(-x.^2/2)/sqrt(2*pi);
Phi = @(x) 0.5*(1 + erf(x/sqrt(2)));

% Hermite polynomials for the CDF Edgeworth expansion
He2 = @(x) x.^2 - 1; 
He3 = @(x) x.^3 - 3*x; 
He5 = @(x) x.^5 - 10*x.^3 + 15*x;

% Edgeworth CDF formula
prob = Phi(z) - phi(z)*(skew_val/6*He2(z) + (kurt_val-3)/24*He3(z) + skew_val^2/72*He5(z));
prob = max(0, min(1, prob));

fprintf('  Probability (Stability): %.4f%%\n', prob*100);
t_mom = toc;
fprintf(' Computation Time: %.4f s\n', t_mom);

%% 7. PLOT ANALYSTICAL EDGEWORTH PDF 
disp('------------------------------------------------');
disp('5. PLOTTING ANALYTICAL DISTRIBUTIONS');

% 1. Define a plotting range based on the calculated moments
plot_min = mean_val - 4*std_dev;
plot_max = mean_val + 4*std_dev;
x_range = linspace(plot_min, plot_max, 1000);

% 2. Standardize the range for Hermite evaluation
z_range = (x_range - mean_val) / std_dev;

% 3. Define Hermite Polynomials for the PDF
He3_pdf = @(x) x.^3 - 3*x;
He4_pdf = @(x) x.^4 - 6*x.^2 + 3;
He6_pdf = @(x) x.^6 - 15*x.^4 + 45*x.^2 - 15;

% 4. Calculate Edgeworth PDF
% Formula: phi(z) * [1 + (skew/6)*He3 + ((kurt-3)/24)*He4 + (skew^2/72)*He6] / std_dev
term_skew = (skew_val/6) * He3_pdf(z_range);
term_kurt = ((kurt_val - 3)/24) * He4_pdf(z_range);
term_skew2 = (skew_val^2/72) * He6_pdf(z_range);

edgeworth_pdf = phi(z_range) .* (1 + term_skew + term_kurt + term_skew2) / std_dev;

% 5. Calculate Standard Normal PDF for comparison
normal_pdf = normpdf(x_range, mean_val, std_dev);

% 6. Create Figure
figure('Color', 'w');
hold on;

% Plot Edgeworth curve
plot(x_range, edgeworth_pdf, 'g-', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Edgeworth PDF (%.1f%% Stable)', prob*100));

% Plot Normal curve
plot(x_range, normal_pdf, 'm:', 'LineWidth', 2, ...
    'DisplayName', sprintf('Normal PDF (%.1f%% Stable)', Phi(z)*100));

% Stability Boundary (x=0)
xline(0, 'k--', 'LineWidth', 2, 'DisplayName', 'Stability Boundary');

% Formatting
grid on; box on;
xlabel('Maximum Real Eigenvalue (\lambda)', 'FontWeight', 'bold');
ylabel('Probability Density');
title('Method 4: Edgeworth & Normal using Multinomial Method ');
legend('Location', 'best');

% Shading the Stable Region 
stable_idx = x_range < 0;
fill([x_range(stable_idx) 0], [edgeworth_pdf(stable_idx) 0], 'g', ...
    'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');

%% ========================================================================
%  SMOLYAK GRID GENERATOR 
%  ========================================================================
function [nodes, weights] = smolyak_grid(dim, level)
    q_min = 1; q_max = level + 1;
    indices = get_indices(dim, q_min, dim + level);
    nodes = []; weights = [];
    for i = 1:size(indices, 1)
        idx = indices(i, :);
        [n_level, w_level] = get_tensor_grid(idx);
        q_sum = sum(idx);
        coeff = (-1)^(dim + level - q_sum) * nchoosek(dim - 1, dim + level - q_sum);
        if coeff ~= 0
            w_level = w_level * coeff;
            if isempty(nodes), nodes = n_level; weights = w_level;
            else, nodes = [nodes; n_level]; weights = [weights; w_level]; end
        end
    end
    nodes_rnd = round(nodes, 8); 
    [~, unique_idx, bin_map] = unique(nodes_rnd, 'rows');
    final_nodes = nodes(unique_idx, :);
    final_weights = zeros(size(final_nodes, 1), 1);
    for i = 1:length(weights)
        final_weights(bin_map(i)) = final_weights(bin_map(i)) + weights(i);
    end
    nodes = final_nodes; weights = final_weights / (2^dim);
end

function idx_list = get_indices(dim, current_val, max_sum)
    if dim == 1, idx_list = (current_val : (max_sum))'; else
        idx_list = [];
        for val = current_val : max_sum - (dim-1)
            rem_list = get_indices(dim-1, 1, max_sum - val);
            new_rows = [repmat(val, size(rem_list,1), 1), rem_list];
            idx_list = [idx_list; new_rows];
        end
    end
end

function [nodes, weights] = get_tensor_grid(idx)
    dim = length(idx);
    nodes_cell = cell(1, dim); weights_cell = cell(1, dim);
    for d = 1:dim
        [n_1d, w_1d] = clenshaw_curtis_1d(idx(d));
        nodes_cell{d} = n_1d; weights_cell{d} = w_1d;
    end
    if dim == 1, nodes = nodes_cell{1}; weights = weights_cell{1}; return; end
    [N{1:dim}] = ndgrid(nodes_cell{:}); [W{1:dim}] = ndgrid(weights_cell{:});
    nodes = zeros(numel(N{1}), dim); weights = W{1};
    for d = 2:dim, weights = weights .* W{d}; end
    weights = weights(:); for d = 1:dim, nodes(:, d) = N{d}(:); end
end

function [x, w] = clenshaw_curtis_1d(level)
    if level == 1, x = 0; w = 2; return; end
    n = 2^(level-1) + 1;
    x = -cos(pi * (0:n-1)' / (n-1)); x(abs(x) < 1e-10) = 0;
    N = n - 1; c = zeros(N+1, 1); c(1:2:N+1) = 2 ./ [1, 1-(2:2:N).^2];
    f = real(ifft([c(1:N+1); c(N:-1:2)]));
    w = [f(1); 2*f(2:N); f(N+1)];
end
