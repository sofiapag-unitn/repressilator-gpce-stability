% ========================================================================
% REPRESSILATOR STABILITY ANALYSIS
% Handles both Symmetric (2D) and Asymmetric (6D) cases, and n=2 vs n=3
% ========================================================================
clear; clc; close all;

addpath(genpath('C:\Users\pagoa\Downloads\PoCET-master\PoCET-master'));
if exist('get_PSImap', 'file') ~= 2
    error('get_PSImap not found: check that the full PoCET toolbox folder (including auxiliary/) is on the path.');
end

%% 1. CONFIGURATION TOGGLES 
disp('================================================');
disp('1. SIMULATION CONFIGURATION');

% TOGGLE 1: Choose the Hill Coefficient 
n_hill = 2;  % Set to 2 (100% stable) or 3 (mixed stability)

% TOGGLE 2: Choose the Model Topology
% true  = 2D Symmetric (Identical parts)
% false = 6D Asymmetric 
run_symmetric = true;  

fprintf('Hill Coefficient (n): %d\n', n_hill);
if run_symmetric
    disp('Topology: 2D SYMMETRIC (Alpha and Delta shared)');
else
    disp('Topology: 6D ASYMMETRIC (Independent parameters)');
end

%% 2. PARAMETER RANGES & gPCE SETUP
disp('------------------------------------------------');
disp('2. PARAMETER RANGES');

if run_symmetric
    % --- 2D SYMMETRIC SETUP ---
    param_names = {'alpha_sq', 'delta'};
    n_params = 2;
    bounds = [0.100, 1.950;   % alpha_sq (shared)
              0.140, 0.700];  % delta (shared)
    %bounds = [0, 3;   % alpha_sq (shared)
              %0, 3];  % delta (shared)
else
    % --- 6D ASYMMETRIC SETUP ---
    param_names = {'alpha1_sq','alpha2_sq','alpha3_sq','delta1','delta2','delta3'};
    n_params = 6;
    bounds = [0.100, 1.950;   % alpha1_sq
              0.100, 1.950;   % alpha2_sq
              0.100, 1.950;   % alpha3_sq
              0.140, 0.700;   % delta1
              0.140, 0.700;   % delta2
              0.140, 0.700];  % delta3
end

disp('Parameter ranges:');
for i = 1:n_params
    fprintf('  %s: [%6.3f, %6.3f]\n', param_names{i}, bounds(i,1), bounds(i,2));
end

disp('------------------------------------------------');
disp('3. gPCE CONFIGURATION');

order = 3;                       % Total polynomial degree
n_xi = n_params;                 % Dimensionality n = # uniform RVs
sys = struct();
sys.pce.options.n_xi = n_xi;
sys.pce.options.order = order;
sys.randomVariables = 1:n_xi;    %tells PoCETbasis which sys.parameters are stochastic

for i = 1:n_xi
    sys.parameters(i).name = param_names{i};
    sys.parameters(i).dist = 'uniform';
end
sys.pce.vars = param_names;
sys.states(1).name = 'dummy';  % λ_max treated as "dummy state"

%% 3. GENERATE BASIS
disp('------------------------------------------------');
disp('4. GENERATING BASIS');

basis_func = PoCETbasis(sys);   % calls the external library to generate orthogonal polynomials
test_args = num2cell(zeros(1, n_xi)); %creates a cell array of zeros to pass into the basis function to test it
basis_test = basis_func(test_args{:});  %evaluates the basis function at the origin
p = numel(basis_test); %counts the total number of polynomial basis terms

% create the anonymous function wrapper 
% it takes the single vector you gave it (xi_vec)
% extracts the first number (xi_vec(1))
% extracts the second number (xi_vec(2))
% feeds them into the basis_func as the comma-separated arguments it demands
if run_symmetric
    basis_func_vec = @(xi_vec) basis_func(xi_vec(1), xi_vec(2));
else
    basis_func_vec = @(xi_vec) basis_func(xi_vec(1), xi_vec(2), xi_vec(3), ...
                                         xi_vec(4), xi_vec(5), xi_vec(6)); 
end
disp(['Basis dimension p = ' num2str(p)]);

%% 4. SAMPLING - GAUSS-LEGENDRE TENSOR PRODUCT QUADRATURE
disp('------------------------------------------------');
disp('5. GAUSS-LEGENDRE QUADRATURE');

% set quadrature points
if run_symmetric
    n_quad_1d = 30;  % 30^2 = 900 points (2D)
else
    n_quad_1d = 4;   % 4^6 = 4,096 points (6D)
end

%returns two column vectors: x (the evaluation points, or "nodes")
% w (the integration weights)
function [x, w] = gauss_legendre_1d(n)
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    T = diag(beta, 1) + diag(beta, -1);  % constructs the Jacobi matrix (T)
    [V, D] = eig(T); 
    %the eigenvalues of the Jacobi matrix are exactly the roots of the n-th
    %Legendre polynomial
    %eigenvalues are our quadrature nodes
    x = diag(D); %extracts the eigenvalues from the diagonal matrix D 
    % into a simple column vector x
    [x, i] = sort(x); %sort(x) orders them from smallest to largest
    %output i captures the original index positions so we can sort
    % the eigenvectors to match 
    w = 2 * V(1, i)'.^2; 
    % the Golub-Welsch theorem states that the weight wj​ for a specific node xj​ 
    % is equal to the integral of the weight function over the interval
    % multiplied by the squared first element of the j-th normalized eigenvector
end

% calls the local function to get 1D nodes and weights
[xi_1d, w_1d] = gauss_legendre_1d(n_quad_1d); 

fprintf('Creating %dD tensor product grid...\n', n_xi);
n_dim = n_xi;
grid_cells = cell(1, n_dim); %prepares a container for the multi-dimensional grid
if run_symmetric
    [grid_cells{:}] = ndgrid(xi_1d, xi_1d); %unpack the empty buckets we made and use them to catch whatever ndgrid throws out
else
    [grid_cells{:}] = ndgrid(xi_1d, xi_1d, xi_1d, xi_1d, xi_1d, xi_1d);
end

% this creates an empty matrix filled with zeros. It has 4096 rows and 6 columns.
xi_samples = zeros(n_quad_1d^n_dim, n_dim); 
% this loop goes through parameters one column at a time
for i = 1:n_dim
    xi_samples(:, i) = grid_cells{i}(:); 
end

%if parameters are independent, the weight of a combined coordinate point
% is just the product of its individual 1D weights
w_nd = w_1d;
for i = 2:n_dim
    w_nd = kron(w_nd, w_1d); 
end
weights = w_nd(:);
weights = weights / (2^n_dim);

N_total = size(xi_samples, 1);
fprintf('Generated %d Gauss-Legendre quadrature points\n', N_total);
fprintf('  Points per dimension: %d\n', n_quad_1d);
fprintf('  Total points: %d^%d = %d\n', n_quad_1d, n_dim, N_total);
fprintf('  Sum of weights: %.6f \n', sum(weights));

%% 5. SYSTEM EVALUATION WITH QUADRATURE WEIGHTS
% this section takes every single point from the grid
% translates it into real biological rates
% calculates the steady state
% and then tests if that settled state is stable

disp('------------------------------------------------');
disp('6. SYSTEM EVALUATION WITH QUADRATURE');

tic_quad_true = tic;
max_eigs = zeros(N_total, 1);

% mutes fzero texts prints
fzero_opts = optimset('Display','off');


for k = 1:N_total
    % grabs the k-th row from the spreadsheet of coordinates
    % this gives us a single vector of numbers between −1 and 1.
    xi = xi_samples(k, :); 
    
    % Transform from standardized ξ to physical parameters
    params = zeros(1, n_xi);
    for i = 1:n_xi
        a = bounds(i, 1); b = bounds(i, 2);
        params(i) = (a + b)/2 + xi(i)*(b - a)/2; 
        %shift the [−1,1] points so they perfectly fit into physical bounds
    end
    
    % assign parameters dynamically based on selected topology
    if run_symmetric
        a1_sq = params(1); a2_sq = params(1); a3_sq = params(1);
        d1 = params(2);    d2 = params(2);    d3 = params(2);
    else
        a1_sq = params(1); a2_sq = params(2); a3_sq = params(3);
        d1 = params(4);    d2 = params(5);    d3 = params(6);
    end
    
    % Define the Hill functions dynamically
    g1 = @(x) a1_sq / (1 + x^n_hill);
    g2 = @(x) a2_sq / (1 + x^n_hill);
    g3 = @(x) a3_sq / (1 + x^n_hill);
    
    % Create the composite map logic for fzero
    map_x1 = @(x3) g1(x3) / d1;
    map_x2 = @(x3) g2(map_x1(x3)) / d2;
    obj_fun = @(x3) x3 - (g3(map_x2(x3)) / d3); 
    
    % Numerically solve for the unique steady state 
    % MATLAB solver searches the range [0,1000] 
    % until it finds the exact concentration of x3​ (x3_star) 
    % that makes obj_fun equal zero
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

%% 6. TRUE MODEL QUADRATURE
% calculates total stability probability by summing the weights 
% of only the stable (eigenvalue < 0) points
prob_stable_qd = sum(weights(max_eigs < 0));  
fprintf('Gauss-Legendre quadrature results:\n');
fprintf('  Stability probability: %.6f%%\n', prob_stable_qd*100);
time_quad_true = toc(tic_quad_true);
fprintf('True model quadrature computation time: %.4f seconds\n', time_quad_true);

%% 7. COMPUTE gPCE COEFFICIENTS
disp('------------------------------------------------');
disp('7. COMPUTING gPCE COEFFICIENTS');
% A is a large design matrix where every row represents one coordinate point
% and every column represents one specific basis function
A = zeros(N_total, p); 
for i = 1:N_total
    %it feeds the specific coordinate point xi_samples(i, :) 
    % into the anonymous function we created 
    % to evaluate all p polynomial terms at that exact location
    A(i, :) = basis_func_vec(xi_samples(i, :))';
end

W = diag(weights);
% we use the least squares solution
% we are solving (AᵀWA)·c = AᵀW·y
coeffs = (A'*W*A) \ (A'*W * max_eigs);
coeffs = coeffs(:);

% check if matrix is well conditioned
fprintf('Condition number: %e\n', cond(A));
fprintf('Constant term (mean): %.6f\n', coeffs(1));

%% 8. MC on Surrogate gPCE
disp('================================================');
disp('8. Performing Monte Carlo on surrogate model');

tic_method2 = tic;
N_surrogate = 500000;  
% generates a massive matrix (500,000 rows by either 2 or 6 columns) 
% of random numbers uniformly distributed between 0 and 1
% we need these random points to fit into [−1,1] 
xi_surrogate = -1 + 2*rand(N_surrogate, n_xi);
surrogate_vals = zeros(N_surrogate, 1);

for i = 1:N_surrogate
    surrogate_vals(i) = basis_func_vec(xi_surrogate(i, :))' * coeffs;
end

% it checks all 500,000 estimated eigenvalues 
% and turns them into 1 (True/Stable) or 0 (False/Unstable)
surrogate_stable = mean(surrogate_vals < 0);
time_method2 = toc(tic_method2);

fprintf('\nMETHOD 2 - gPCE Surrogate Monte Carlo:\n');
fprintf('  Surrogate stability: %.4f%% (N=%d)\n', surrogate_stable*100, N_surrogate);
fprintf('  Computation time: %.4f seconds\n', time_method2);

%% 9. MOMENT COMPUTATION
disp('================================================');
disp('8. MOMENT COMPUTATION ');
tic_method3 = tic;
% generate the surrogate model's estimated eigenvalues 
% at all of those specific grid
surrogate_quad_vals = A * coeffs;  
mean_val = sum(weights .* surrogate_quad_vals);
second_moment = sum(weights .* surrogate_quad_vals.^2);
third_moment = sum(weights .* surrogate_quad_vals.^3);
fourth_moment = sum(weights .* surrogate_quad_vals.^4);

% standard algebraic expansion formulas to convert the raw moments into central moments
central_moment2 = second_moment - mean_val^2;
central_moment3 = third_moment - 3*mean_val*second_moment + 2*mean_val^3;
central_moment4 = fourth_moment - 4*mean_val*third_moment + ...
                  6*mean_val^2*second_moment - 3*mean_val^4;

std_dev = sqrt(central_moment2);
if std_dev > 0
    skewness_val = central_moment3 / (std_dev^3);
    kurtosis_val = central_moment4 / (std_dev^4);
else
    skewness_val = 0;
    kurtosis_val = 3;
end

fprintf('\nMoments from Gauss-Legendre quadrature:\n');
fprintf('  Mean: %.6f\n', mean_val);
fprintf('  Std Dev: %.6f\n', std_dev);
fprintf('  Skewness: %.6f\n', skewness_val);
fprintf('  Kurtosis: %.6f\n', kurtosis_val);

%% 10. EDGEWORTH EXPANSION WITH QUADRATURE MOMENTS
disp('================================================');
disp('9. EDGEWORTH EXPANSION WITH QUADRATURE MOMENTS');

z = -mean_val / std_dev;
% standard normal pdf
phi = @(x) exp(-x.^2/2) / sqrt(2*pi);
% standard normalcdf
Phi = @(x) 0.5 * (1 + erf(x/sqrt(2)));

He2 = @(x) x.^2 - 1;
He3 = @(x) x.^3 - 3*x;
He4 = @(x) x.^4 - 6*x.^2 + 3;
He5 = @(x) x.^5 - 10*x.^3 + 15*x;
He6 = @(x) x.^6 - 15*x.^4 + 45*x.^2 - 15;

edgeworth_cdf = Phi(z) - phi(z) * ( ...
    skewness_val/6 * He2(z) + ...
    (kurtosis_val-3)/24 * He3(z) + ...
    skewness_val^2/72 * He5(z) );

normal_cdf = Phi(z);
edgeworth_cdf = max(0, min(1, edgeworth_cdf));
normal_cdf = max(0, min(1, normal_cdf));

time_method3 = toc(tic_method3);
fprintf('\nMETHOD 3 - gPCE + Edgeworth:\n');
fprintf('  Standardized threshold z = (0 - mean)/std = %.4f\n', z);
fprintf('  Normal approximation:    %.6f%%\n', normal_cdf*100);
fprintf('  Edgeworth expansion:     %.6f%%\n', edgeworth_cdf*100);
fprintf('  Computation time: %.4f seconds\n', time_method3);

%% 11. VALIDATION - REFERENCE MONTE CARLO
disp('================================================');
disp('10. REFERENCE MONTE CARLO (TRUE MODEL)');

tic_method1 = tic;
N_validate = 1000000; 
validate_vals = zeros(N_validate, 1);
validate_stable = 0;

for i = 1:N_validate
    xi_val = -1 + 2*rand(1, n_xi);
    
    params = zeros(1, n_xi);
    for j = 1:n_xi
        a = bounds(j, 1); b = bounds(j, 2);
        params(j) = (a + b)/2 + xi_val(j)*(b - a)/2;
    end
    
    % Assign parameters dynamically
    if run_symmetric
        a1_sq = params(1); a2_sq = params(1); a3_sq = params(1);
        d1 = params(2);    d2 = params(2);    d3 = params(2);
    else
        a1_sq = params(1); a2_sq = params(2); a3_sq = params(3);
        d1 = params(4);    d2 = params(5);    d3 = params(6);
    end
    
    g1 = @(x) a1_sq / (1 + x^n_hill);
    g2 = @(x) a2_sq / (1 + x^n_hill);
    g3 = @(x) a3_sq / (1 + x^n_hill);
    
    map_x1 = @(x3) g1(x3) / d1;
    map_x2 = @(x3) g2(map_x1(x3)) / d2;
    obj_fun = @(x3) x3 - (g3(map_x2(x3)) / d3);
    
    % Solve for steady state 
    x3_star = fzero(obj_fun, [0, 1000], fzero_opts);
    x1_star = map_x1(x3_star);
    x2_star = map_x2(x3_star);
    
    % Evaluate dependent Jacobian slopes
    k1 = (n_hill * a1_sq * x3_star^(n_hill-1)) / ((1 + x3_star^n_hill)^2);
    k2 = (n_hill * a2_sq * x1_star^(n_hill-1)) / ((1 + x1_star^n_hill)^2);
    k3 = (n_hill * a3_sq * x2_star^(n_hill-1)) / ((1 + x2_star^n_hill)^2);
    
    J = [-d1, 0, -k1;
         -k2, -d2, 0;
         0, -k3, -d3];
    
    max_eig = max(real(eig(J)));
    validate_vals(i) = max_eig;
    
    if max_eig < 0
        validate_stable = validate_stable + 1;
    end
end
prob_stable_validate = validate_stable / N_validate;

time_method1 = toc(tic_method1);
fprintf('\nMETHOD 1 - Reference Monte Carlo (true model):\n');
fprintf('  Monte Carlo stability: %.6f%% (N=%d)\n', prob_stable_validate*100, N_validate);
fprintf('  Computation time: %.4f seconds\n', time_method1);

%% 12. COMPREHENSIVE COMPARISON OF ALL METHODS
disp('================================================');
disp('11. COMPARISON OF STABILITY PROBABILITY ESTIMATES');
disp('================================================');

fprintf('\n%-45s %12s %12s %15s\n', 'Method', 'Probability', 'Error', 'Time (s)');
fprintf('%-45s %12s %12s %15s\n', '------', '-----------', '-----', '--------');

fprintf('%-45s %11.4f%% %12s %15.4f\n', ...
    sprintf('1. Reference MC (true model, %d samples)', N_validate), ...
    prob_stable_validate*100, '0.0000%', time_method1);

error_quad_true = (prob_stable_qd - prob_stable_validate) * 100;
fprintf('%-45s %11.4f%% %+11.4f%% %15.4f\n', ...
    sprintf('2. True model + Quadrature (%d pts)', N_total), ...
    prob_stable_qd*100, error_quad_true, time_quad_true);  

error_surrogate_mc = (surrogate_stable - prob_stable_validate) * 100;
fprintf('%-45s %11.4f%% %+11.4f%% %15.4f\n', ...
    '3. gPCE Surrogate + MC (500k samples)', ...
    surrogate_stable*100, error_surrogate_mc, time_method2);

tic_method4 = tic;
surrogate_stable_quad = sum(weights(surrogate_quad_vals < 0));
time_method4 = toc(tic_method4);
error_surrogate_quad = (surrogate_stable_quad - prob_stable_validate) * 100;
fprintf('%-45s %11.4f%% %+11.4f%% %15.4f\n', ...
    sprintf('4. gPCE Surrogate + Quadrature (%d pts)', N_total), ...
    surrogate_stable_quad*100, error_surrogate_quad, time_method4);

error_edgeworth = (edgeworth_cdf - prob_stable_validate) * 100;
fprintf('%-45s %11.4f%% %+11.4f%% %15.4f\n', ...
    '5. gPCE + Edgeworth (no sampling)', ...
    edgeworth_cdf*100, error_edgeworth, time_method3);

error_normal = (normal_cdf - prob_stable_validate) * 100;
fprintf('%-45s %11.4f%% %+11.4f%% %15s\n', ...
    '6. Normal approximation (for reference)', ...
    normal_cdf*100, error_normal, '-');

%% 13. PLOTTING - THREE SEPARATE COMPARISON PLOTS
figure('Color','w', 'Position', [100, 100, 1800, 500]);

% Dynamically calculate a universal X-axis with padding
global_min = min([min(validate_vals), min(surrogate_vals)]) - 0.2;
global_max = max([max(validate_vals), max(surrogate_vals)]) + 0.2;
univ_x_range = linspace(global_min, global_max, 1000);
num_bins = 50; % Define bin count for consistency

% --- PLOT 1: Reference Monte Carlo  ---
subplot(1,3,1);
hold on;
% Plot the histogram 
h1 = histogram(validate_vals, num_bins, 'Normalization', 'pdf', ...
    'FaceColor', [0.7, 0.7, 0.7], 'EdgeColor', 'w', 'DisplayName', 'Reference MC');

xline(0, 'k--', 'LineWidth', 2, 'DisplayName', 'Stability Boundary');

% we use vertical colored patches in the background to indicate zones.
ylim_val = [0, max(h1.Values)*1.2];
patch([global_min 0 0 global_min], [0 0 ylim_val(2) ylim_val(2)], [0.2, 0.8, 0.2], ...
    'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', sprintf('Stable Zone (%.1f%%)', prob_stable_validate*100));
patch([0 global_max global_max 0], [0 0 ylim_val(2) ylim_val(2)], [0.8, 0.2, 0.2], ...
    'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', sprintf('Unstable Zone (%.1f%%)', (1-prob_stable_validate)*100));

mean_ref = mean(validate_vals);
xline(mean_ref, 'b:', 'LineWidth', 2, 'DisplayName', sprintf('Mean = %.3f', mean_ref));

xlabel('Maximum Real Eigenvalue (\lambda)', 'FontWeight', 'bold');
ylabel('Probability Density', 'FontWeight', 'bold');
title({'Method 1: Reference Monte Carlo', sprintf('Stability: %.2f%%', prob_stable_validate*100)}, 'FontWeight', 'bold');
legend('Location','best');
grid on; box on; xlim([global_min, global_max]); ylim(ylim_val);

% --- PLOT 2: Method 2 - gPCE Surrogate vs Reference MC ---
subplot(1,3,2);
hold on;
% Reference Histogram
histogram(validate_vals, num_bins, 'Normalization', 'pdf', ...
    'FaceColor', 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
    'DisplayName', sprintf('Reference MC (%.1f%%)', prob_stable_validate*100));

% Surrogate Histogram
histogram(surrogate_vals, num_bins, 'Normalization', 'pdf', ...
    'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
    'DisplayName', sprintf('gPCE Surrogate (%.1f%%)', surrogate_stable*100));

xline(0, 'k--', 'LineWidth', 2, 'DisplayName', 'Stability Boundary');

xlabel('Maximum Real Eigenvalue (\lambda)', 'FontWeight', 'bold');
ylabel('Probability Density', 'FontWeight', 'bold');
title({'Method 2: gPCE Surrogate vs Reference', ...
       sprintf('Error: %+.2f%%', (surrogate_stable - prob_stable_validate)*100)}, 'FontWeight', 'bold');
legend('Location','best');
grid on; box on; xlim([global_min, global_max]);

% --- PLOT 3: Method 3 - Edgeworth Expansion vs Reference MC 
subplot(1,3,3);
hold on;

% 1. Plot the Reference Histogram 
h_ref = histogram(validate_vals, 100, 'Normalization', 'pdf', ...
    'FaceColor', [0.85, 0.85, 0.85], 'EdgeColor', 'none', ...
    'DisplayName', 'Ref MC Histogram');

% 2. Calculate the Analytical Curves over the universal range
z_range = (univ_x_range - mean_val) / std_dev;

% Edgeworth PDF
% It uses He3, He4, and He6 because d/dz [phi(z)He_n] = -phi(z)He_{n+1}
edgeworth_pdf = phi(z_range) .* (1 + ...
    (skewness_val/6) * He3(z_range) + ...
    ((kurtosis_val-3)/24) * He4(z_range) + ...
    (skewness_val^2/72) * He6(z_range)) / std_dev;

% Normal PDF
normal_pdf = normpdf(univ_x_range, mean_val, std_dev);

% 3. Plot the Curves
plot(univ_x_range, edgeworth_pdf, 'g-.', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Edgeworth (%.1f%%)', edgeworth_cdf*100));

plot(univ_x_range, normal_pdf, 'm:', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Normal (%.1f%%)', normal_cdf*100));

% 4. Formatting and Boundary
xline(0, 'k--', 'LineWidth', 2, 'DisplayName', 'Stability Boundary');

xlabel('Maximum Real Eigenvalue (\lambda)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Probability Density', 'FontSize', 11, 'FontWeight', 'bold');

title({'Method 3: Edgeworth & Normal vs Reference', ...
       sprintf('Edgeworth Error: %+.2f%% | Normal Error: %+.2f%%', ...
       (edgeworth_cdf - prob_stable_validate)*100, (normal_cdf - prob_stable_validate)*100)}, ...
      'FontSize', 12, 'FontWeight', 'bold');

legend('Location','best', 'FontSize', 9);
grid on; box on;
xlim([global_min, global_max]);

% Ensure the Y-axis accommodates both the histogram and the curves
ylim([0, max([max(h_ref.Values), max(edgeworth_pdf), max(normal_pdf)]) * 1.1]);
% Global Title
sgtitle(sprintf('Comparison of Methods for Stability Analysis (n=%d)', n_hill), 'FontWeight', 'bold');
figure('Color','w', 'Position', [100, 100, 1800, 500]);

