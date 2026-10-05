function MI = generate_multi_indices(dims, max_order)
    % Generates Total Order multi-indices 
    % Recursively finds all alpha such that sum(alpha) <= max_order
    MI = get_mi_recursive(dims, max_order);
    
    sums = sum(MI, 2);% sum the rows
    [~, sort_idx] = sortrows([sums, MI]); % Sort by sum, then indices
    MI = MI(sort_idx, :);
end