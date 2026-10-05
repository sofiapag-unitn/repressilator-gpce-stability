
function coeff = multinomial_coeff(k)
    % k is a row vector [k1 ... kp] with sum(k) = m
    m = sum(k);
    coeff = factorial(m) / prod(factorial(k));
end




