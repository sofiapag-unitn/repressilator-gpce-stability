function [x, w] = gauss_legendre_1d(n)
    % Standard Gauss-Legendre nodes and weights on [-1, 1]
    % Returns weights normalized to sum to 1 (for expectation E[])
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2)); %This calculates the coefficients for the matrix (T)
    T = diag(beta, 1) + diag(beta, -1); %This creates a symmetric tridiagonal matrix T of size n×n.
    [V, D] = eig(T); 
    % The eigenvalues of this matrix T are exactly the roots of the n-th degree Legendre polynomial (the nodes).
    x = diag(D); [x, i] = sort(x);
    % diag(D) pulls the eigenvalues out of the diagonal matrix D and stores them in x.
    % sort(x) ensures the nodes are in increasing order (e.g., from -1 toward 1). 
    % The index vector i tracks how the nodes were moved so the weights can be reordered the same way.
    w = 2 * V(1, i)'.^2;
    w = w(:) / 2; % Normalize for Probability Measure (Uniform = 1/2)
end
