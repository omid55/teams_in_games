%Omid55
function [config] = convex_optimization_problem(fname, iskernel, Cs, sigma)
    %fname = 'small_winner_minus_loser.csv';
    %fname = 'small_winner_minus_loser_close.csv';

    X = csvread(fname,1,0);

    train_size = round(0.6 * size(X,1));
    validation_size = round(0.1 * size(X,1));
    
    idx = randperm(size(X,1));
    train = X(idx(1:train_size),:);
    validation = X(idx(train_size+1:train_size+validation_size),:);
    test = X(idx(train_size+validation_size+1:end),:);
    
%     POS_TRAIN = 100*length(find(sum(train')>0)) / size(train,1)
%     POS_VALID = 100*length(find(sum(validation')>0)) / size(validation,1)
%     POS_TEST = 100*length(find(sum(test')>0)) / size(test,1)

    % preprocessing
    config.CS = Cs;
    config.WS = zeros(length(config.CS), 5);
    config.SV = zeros(length(config.CS), 1);
    config.Train_acc = zeros(length(config.CS), 1);
    config.Valid_acc = zeros(length(config.CS), 1);
    
    if iskernel
        % RBF
        config.sigma = sigma;
        Z = apply_RBF(train, config.sigma);
    else
        Z = train*train';
    end
    
    for i = 1 : length(config.CS)
        Z = Z + diag(ones(train_size,1)*1/(2*config.CS(i)));
        
        % optimization
        cvx_begin
            variable a(train_size)
            minimize( 0.5 * a' * Z * a - sum(a) );
            subject to
                a >= 0
        cvx_end

        config.SV(i) = length(find(a~=0));

        w = a' * train;
        config.WS(i,:) = w;

        r = w * train';
        config.Train_acc(i) = 100 * (length(find(r>0))/length(r));

        r = w * validation';
        config.Valid_acc(i) = 100 * (length(find(r>0))/length(r));
    end

    config.selected = find(config.Valid_acc == max(config.Valid_acc));
    if(length(config.selected)>1)
        config.selected = config.selected(1);
    end
    best_w = config.WS(config.selected,:);
    r = best_w * test';
    config.Test_acc = 100 * (length(find(r>0))/length(r));
    
    rand_acc = 0;
    rn = 1000;
    for i = 1 : rn 
        rand_w = 2*rand(1,5)-1;
        r = rand_w*test';
        rand_acc = rand_acc + 100 * (length(find(r>0))/length(r));
    end
    rand_acc = rand_acc / rn;
    
    config.Random_Test_acc = rand_acc;
end


function [new_data] = apply_RBF(data, sigma)
    nsq = sum(data.^2,2);
    K = bsxfun(@minus,nsq,(2*data)*data.');
    K = bsxfun(@plus,nsq.',K);
    new_data = exp( -K/(2*sigma^2) );
end

