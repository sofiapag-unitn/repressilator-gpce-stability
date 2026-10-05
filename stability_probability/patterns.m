function K = patterns(p,m)
%the total number of unique patterns is (m+p-1) choose m
num_patterns = nchoosek(m+p-1,m);
K = zeros(num_patterns,p);

%Initialize k
k= zeros(1,p);
k(p) = m; % all items in the last position

row=1;
K(row, :) = k;

while k(1) < m 
% Start on the last position, look for the non empty position
 t = p; 
 while t>=1 && k(t)==0
    t = t-1;
 end 

 if t == p
    % if non empty position is the last position then PUSH
    k(p-1) = k(p - 1) + 1;
    k(p) = k(p) - 1;

 else
    % then FORK
    k(t-1) = k(t-1) + 1;
    k(p) = k(t) - 1 ;
    k(t) = 0;
 end

 row = row + 1;
 K(row, :) = k;
end 

fprintf('Generated %d unique patterns for p=%d, m=%d: \n',row,p,m);
%disp(K);

