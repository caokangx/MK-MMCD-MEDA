% DEMO for testing MEDA on PIE datasets
str_domains = {'05', '07', '09', '27', '29'};
list_acc = [];
% gamma
gamma_list = [0.6];

options.d = 20;
options.rho = 1.0;
options.p = 10;
options.lambda = 10.0;
options.eta = 0.1;
options.T = 10;
options.gamma = 0.2;
options.mu = 0.6;

fid = fopen(strcat('./result_PIE.txt'),'at');

for gamma = gamma_list
    options.gamma = gamma;
    fprintf(fid,'gamma = %f\n', gamma);
    
    for i = 1 : 5
        for j = 1 : 5
            if i == j
                continue;
            end
            src = str_domains{i};
            tar = str_domains{j};
            
            load(['data/PIE/PIE' src '.mat']);
            
            % source domain
            fea = fea ./ repmat(sum(fea, 2), 1, size(fea,2));
            Xs = zscore(fea, 1);   clear fea
            Ys = gnd;              clear gnd
            
            load(['data/PIE/PIE' tar '.mat']);
            
            % target domain
            fea = fea ./ repmat(sum(fea, 2), 1, size(fea,2));
            Xt = zscore(fea, 1);   clear fea
            Yt = gnd;              clear gnd
            
            % meda
            %         [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
            [Acc,~,~,~] = MK_MMCD(Xs,Ys,Xt,Yt,options);
            fprintf('%s --> %s: %.2f accuracy \n\n', src, tar, Acc * 100);
            list_acc = [list_acc; Acc];
            
            fprintf(fid, '%f\n', Acc * 100); 
        end
    end
end

fclose(fid);
