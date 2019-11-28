function [Acc,acc_iter,Beta,Yt_pred] = MK_MMCD(Xs,Ys,Xt,Yt,options)

% Reference:
%% Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S.
%% Yu. Visual Domain Adaptation with Manifold Embedded Distribution
%% Alignment. ACM Multimedia conference 2018.

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

%% Load algorithm options
addpath(genpath('liblinear/matlab'));

%% Algorithm starts here
    fprintf('MEDA starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end
    if ~isfield(options,'gamma')
        options.gamma = 0.4;
    end

    % Manifold feature learning
     [Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
     Xs = double(Xs_new'); %行变列
     Xt = double(Xt_new'); %行变列


%     % 去掉GFK效果并不好
%     Xs = Xs';
%     Xt = Xt';
    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    C = length(unique(Ys)); %类数量
    acc_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W)))); % D的-1/2次方，这么写计算的快
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

    % Generate soft labels for the target domain
    % KNN分类器
%      knn_model = fitcknn(X(:,1:n)',Ys,'NumNeighbors',1);
%      Cls = knn_model.predict(X(:,n + 1:end)'); % predict Xt
     
    % SVM 分类器 
%      svm_model =  svmtrain(Ys,X(:,1:n)','-t 2 -c 1 -g 0.95');
%      [Cls,accuracy,~] = svmpredict(Yt,X(:,n + 1:end)',svm_model);
    
    % Linear 线性分类器     
      model = train(Ys,sparse(X(:,1:n)'),'-s 2 -c 0.7 -q 1');
      [Cls,~] = predict(Yt,sparse(X(:,n+1:end)'),model);
   
%     % Neural Network分类器
%      %特征值归一化
%     [input,minI,maxI] = premnmx(X(:,1:n))  ;
%     %构造输出矩阵
%     s = length(Ys) ;
%     output = zeros(s,2) ;
%     for i = 1 : s 
%        output(i,Ys(i)) = 1 ;
%     end
%     %创建神经网络
%     net = newff(minmax(input),[50 30 10],{'logsig' 'logsig' 'purelin'},'traingdx') ; 
%     %设置训练?数
%     net.trainparam.show = 50 ;% 显示中间结果的周期
%     net.trainparam.epochs = 500 ;%最大迭代次数（学习次数）
%     net.trainparam.goal = 0.01 ;%神经网络训练的目标误差
%     net.trainParam.lr = 0.01 ;%学习速率（Learning rate）
%     %开始训练
%     %其中input为训练集的输入信号，对应output为训练集的输出结果
%     net = train(net,input,output');
%     
%     %测试数据归一化
%     testInput = tramnmx(X(:,n+1:end),minI,maxI) ;
%     %[testInput,minI,maxI] = premnmx( [t1 , t2]')  ;
%     %仿真
%     %其中net为训练后得到的网络，返回的Y为
%     Y = sim(net,testInput);
%      %统计识别正确率
%      Cls = []
%     [s1,s2] = size(Y) ;
%     for i = 1 : s2
%         [~,index] = max(Y(:,i))
%         Cls = [Cls;index];
%     end

    % Construct kernel
    K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));

    % multi kernel
%     K = multi_kernel(X, sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
    
    E = diag(sparse([ones(n,1);zeros(m,1)]));

    for t = 1 : options.T
        % Estimate mu
        mu = estimate_mu(Xs',Ys,Xt',Cls);
%         mu = options.mu;
        % Construct MMD matrix
        e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
        M0 = e * e' * length(unique(Ys));
        Mc = 0;
        % Z0
        Z = [-1/n^2 * ones(n) + diag(1/n * ones(n,1)), zeros(n,m);
             zeros(m,n)                              , 1/m^2 * ones(m) + diag(-1/m * ones(m,1))];

        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = zeros(n + m,1);
            e(Ys == c) = 1 / length(find(Ys == c));  % 1 / Nsc
            e(n + find(Cls == c)) = -1 / length(find(Cls == c)); % - 1 / Ntc
            e(isinf(e)) = 0;
            Mc = Mc + e * e';
            
            Nsc = length(find(Ys == c));
            Ntc = length(find(Cls == c));
            
            Ys_logical_matrix = (Ys == c) * (Ys == c)';
            Yt_logical_matrix = (Cls == c) * (Cls == c)';

            Zc = [Ys_logical_matrix .* (-1/Nsc^2 * ones(n) + diag(1/Nsc * ones(n,1))), zeros(n,m);
                  zeros(m,n)                                                         ,Yt_logical_matrix .* (1/Ntc^2 * ones(m) + diag(-1/Ntc * ones(m,1)))];
            Zc(isnan(Zc)) = 0;
            Z = Z + Zc;
        end
        M = (1 - mu) * M0 + mu * Mc;
        V = M + options.gamma * Z;
        % norm function has bug, so we compute another way
        V = V / sqrt(sumsqr(V));
        
        % Multi Kernel MMCD
        Beta = ((E + options.lambda * V + options.rho * L ) * K + options.eta * speye(n + m,n + m)) \ (E * YY);

        F = K * Beta;
        [~,Cls] = max(F,[],2); % predict Xt

        %% Compute accuracy
        Acc = numel(find(Cls(n+1:end)==Yt)) / m;
        Cls = Cls(n+1:end);
        acc_iter = [acc_iter;Acc];
        fprintf('Iteration:[%02d]>>mu=%.2f,Acc=%f\n',t,mu,Acc);
    end
    Yt_pred = Cls;
    Acc = max(acc_iter);
    fprintf('MEDA ends!\n');
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
    
end