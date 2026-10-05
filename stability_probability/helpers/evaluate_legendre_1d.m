function P = evaluate_legendre_1d(x, order)
    % Evaluates Legendre polynomials at x
    % P: The output matrix where each column represents a polynomial degree.
    P = zeros(length(x), order+1);
    P(:, 1) = 1;      % P_0
    if order >= 1, P(:, 2) = x; end % P_1
    for k = 2:order
        n = k-1;
        P(:, k+1) = ((2*n+1)*x .* P(:, k) - n*P(:, k-1)) / (n+1);
    end
end
