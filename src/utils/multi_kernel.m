function MK = multi_kernel(X, Sigma)
    % get linear kernel
    linear_kernel_instance = linear_kernel(X);
    % get rbf kernel
    rbf_kernel_instance = rbf_kernel(X, Sigma);
    %  get sam kernel
    sam_kernel_instance = sam_kernel(X, Sigma);
    
    % TODO: get beta according to equation in DAN
    beta = [0, 0.1, 0.9];
    
    MK = beta(1) * linear_kernel_instance + beta(2) * rbf_kernel_instance + beta(3) * sam_kernel_instance;
end

function K = linear_kernel(X)
    K = X' * X;
end

function K = rbf_kernel(X, sigma)
    n1sq = sum(X.^2,1);
    n1 = size(X,2);
    D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
    K = exp(-D/(2*sigma^2));
end

function K = sam_kernel(X, sigma)
    D = X'*X;
    K = exp(-acos(D).^2/(2*sigma^2));
end

