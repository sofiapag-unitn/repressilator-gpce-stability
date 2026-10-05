function list = get_mi_recursive(dims, limit)
    if dims == 1
        list = (0:limit)';
    else
        list = [];
        for i = 0:limit
            sub_list = get_mi_recursive(dims-1, limit-i);
            new_rows = [repmat(i, size(sub_list,1), 1), sub_list];
            list = [list; new_rows];
        end
    end
end
