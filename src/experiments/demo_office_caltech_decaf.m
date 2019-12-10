% DEMO for testing MEDA on Office+Caltech10 datasets
str_domains = {'caltech', 'amazon', 'webcam', 'dslr'};
list_acc = [];
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};

        load(['data/caltech_office31_decaf6/' src '_decaf.mat']);     % source domain
        feas = feas ./ repmat(sum(feas,2),1,size(feas,2)); %每一维度做均值
        Xs = zscore(feas,1);    clear feas   %标准化（归一化）
        Ys = labels;           clear labels
        
        load(['data/caltech_office31_decaf6/' tgt '_decaf.mat']);     % target domain
        feas = feas ./ repmat(sum(feas,2),1,size(feas,2)); %每一维度做均值
        Xt = zscore(feas,1);     clear feas %标准化（归一化）
        Yt = labels;            clear labels
        
        % meda
        options.d = 20;
        options.rho = 1.0;
        options.p = 10;
        options.lambda = 10.0;
        options.eta = 0.1;
        options.T = 10;
        options.gamma = 0.15;
        options.mu = 1;
%         [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        [Acc,~,~,~] = MK_MMCD(Xs,Ys,Xt,Yt,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
        list_acc = [list_acc; Acc];
    end
end
