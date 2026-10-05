function mu_m = multinomial_moment(coeffs, m, MomMats)
    % Access precomputed data for order m
    K = MomMats(m).K;
    bk = MomMats(m).val;
    M_k = MomMats(m).M_k;
    
    coeffs = coeffs(:).'; % Ensure row
    
    % For each row in K, compute prod(coeffs.^k)
    num_patterns = size(K, 1);
    c_powers = zeros(num_patterns, 1);
    for r = 1:num_patterns
        c_powers(r) = prod(coeffs.^K(r, :));
    end
    
    % Final summation 
    mu_m = sum(M_k .* bk .* c_powers);
end
