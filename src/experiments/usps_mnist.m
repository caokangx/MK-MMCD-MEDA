% DEMO for testing MEDA on MNIST and USPS datasets
str_domains = {'MNIST', 'USPS'};
list_acc = [];
for i = 1 : 2
    for j = 1 : 2
        if i == j
            continue;
        end
        src = str_domains{i};
        tar = str_domains{j};
        
        load(['data/' src '_vs_' tar '.mat']);     
        
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
        options.rho = 1.0;
        options.p = 10;
        options.lambda = 10.0;
        options.eta = 0.1;
        options.T = 20;
        options.gamma = 0.1;
%         options.mu = 0.6;
%         [Acc,~,~,~] = MEDA(Xs',Ys,Xt',Yt,options);
        [Acc,~,~,~] = MK_MMCD(Xs',Ys,Xt',Yt,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tar, Acc * 100);
        list_acc = [list_acc; Acc];
    end
end
