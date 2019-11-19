% DEMO for testing MEDA on Office+Caltech10 datasets
str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};
list_acc = [];
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
%         src = 'Caltech10';
%         tgt = 'webcam';
        load(['data/' src '_SURF_L10.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); %每一维度做均值
        Xs = zscore(fts,1);    clear fts   %标准化（归一化）
        Ys = labels;           clear labels
        
        load(['data/' tgt '_SURF_L10.mat']);     % target domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); %每一维度做均值
        Xt = zscore(fts,1);     clear fts %标准化（归一化）
        Yt = labels;            clear labels
        
        % meda
        options.d = 20;
        options.rho = 1.0;
        options.p = 10;
        options.lambda = 10.0;
        options.eta = 0.1;
        options.T = 10;
        options.gamma = 0.1;
        options.mu = 0.6;
%         [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        [Acc,~,~,~] = MK_MMCD(Xs,Ys,Xt,Yt,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
        list_acc = [list_acc; Acc];
    end
end
