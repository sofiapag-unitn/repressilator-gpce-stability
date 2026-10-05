function bk = compute_bk_exact(k_pattern, Data)
% COMPUTE_BK_EXACT Calculates E[Psi^k] using 1D Tensor Product Quadrature
% This is mathematically exact and much faster than grid summation.
    
    MIdx = Data.MultiIndices;   % [p x dims]
    Leg1D = Data.Legendre1D;    % [N_1d x order+1] The values of Legendre polynomials at the Gaussian nodes
    W1D   = Data.Weights1D;     % [N_1d x 1] The Gaussian weights.
    dims = size(MIdx, 2);
    
    % Find which basis functions are active (power > 0)
    active_idx = find(k_pattern > 0); 
    % Example: we only care about polynomial #1 and polynomial #3. Ignore #2 and #4 because their exponent is zero
    if isempty(active_idx), bk = 1; return; end
    
    total_integral = 1;
    
    % Loop over dimensions (Separation of Variables)
    for d = 1:dims
        % Construct the 1D polynomial for dimension 'd'
        % It is the product of Legendre polys: Product( L_{degree_i}(x)^k_i )
        
        poly_vals_1d = ones(size(W1D));
        
        for i = active_idx
            deg = MIdx(i, d);       % Degree of basis function i in dim d
            pow = k_pattern(i);     % Power from multinomial expansion
            
            % Look up precomputed Legendre values (deg+1 because index 1 is deg 0)
            poly_vals_1d = poly_vals_1d .* (Leg1D(:, deg+1) .^ pow);
        end
        
        % Integrate in 1D: sum( w * f(x) )
        integral_1d = sum(W1D .* poly_vals_1d);
        
        total_integral = total_integral * integral_1d;
    end
    
    bk = total_integral;
end