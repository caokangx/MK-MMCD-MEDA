% DEMO for testing MEDA on MNIST and USPS datasets
str_domains = {'1', '2'};
list_acc = [];
for i = 1 : 2
    src = str_domains{i};

    load(['data/COL20/COIL_' src '.mat']);     

    % source domain
    X_src = X_src ./ repmat(sum(X_src, 1), size(X_src,1),1);
    Xs = zscore(X_src, 1); clear X_src
    Ys = Y_src;            clear Y_src

    % target domain
    X_tar = X_tar ./ repmat(sum(X_tar, 1), size(X_tar,1),1);
    Xt = zscore(X_tar, 1); clear X_tar
    Yt = Y_tar;            clear Y_tar

    % meda
    options.d = 20;
    options.rho = 0.1;
    options.p = 10;
    options.lambda = 10.0;
    options.eta = 0.1;
    options.T = 15;
    options.gamma = 0.1;
    options.mu = 0.6;
    options.delta = 0.01;
%     [Acc,~,~,~] = MEDA(Xs',Ys,Xt',Yt,options);
    [Acc,~,~,~] = MK_MMCD(Xs',Ys,Xt',Yt,options);
    fprintf('COIL_%s %.2f accuracy \n\n', src, Acc * 100);
    list_acc = [list_acc; Acc];
end
