function MomMats = precompute_MomMats(n_params, order, max_m)
    % 1. Setup Basis Data 
    ExactData.MultiIndices = generate_multi_indices(n_params, order);
    p = size(ExactData.MultiIndices, 1);
    
    % Use Gauss-Legendre quadrature for exact 1D integrals
    [Nodes1D, Weights1D] = gauss_legendre_1d(10); 
    ExactData.Weights1D = Weights1D;
    ExactData.Legendre1D = evaluate_legendre_1d(Nodes1D, order); 

    MomMats = struct('val', cell(max_m, 1), 'K', cell(max_m, 1), 'M_k', cell(max_m, 1));

    for m = 1:max_m
        fprintf('Precomputing Moment Matrix for m=%d...\n', m);
        
        % Generate unique multinomial patterns (The set K) 
        K = patterns(p, m); 
        num_patterns = size(K, 1);
        
        vals = zeros(num_patterns, 1);
        m_coeffs = zeros(num_patterns, 1);
        
        for r = 1:num_patterns
            k_pattern = K(r, :);
            % Compute exact integral E[Psi^k] 
            vals(r) = compute_bk_exact(k_pattern, ExactData);
            % Compute multiplicity (multinomial coefficient) 
            m_coeffs(r) = multinomial_coeff(k_pattern);
        end
        
        % Filter out terms where the integral is effectively zero
        valid = abs(vals) > 1e-14;
        MomMats(m).val = vals(valid);
        MomMats(m).K = K(valid, :);
        MomMats(m).M_k = m_coeffs(valid);
        
        fprintf('  Stored %d non-zero patterns.\n', sum(valid));
    end
end
