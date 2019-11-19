% n = 6;
% m = 4;
% 
% Ys = [1,1,1,3,3,6];
% 
% Cls = [1,3,3,3];
% 
% c = 3;
% 
% Nsc = length(find(Ys == c));
% Ntc = length(find(Cls == c));
% 
% Ys_logical_matrix = (Ys == c)' * (Ys == c);
% 
% Yt_logical_matrix = (Cls == c)' * (Cls == c);
% 
% Zc = [Ys_logical_matrix .* (-1/Nsc^2 * ones(n) + diag(1/Nsc * ones(n,1))), zeros(n,m);
%       zeros(m,n)                                                         ,Yt_logical_matrix .* (1/Ntc^2 * ones(m) + diag(-1/Ntc * ones(m,1)))];

a = [1 2 3; 4 5 6];

b = sumsqr(a);
b = sqrt(b);
c = norm(a, 'fro');