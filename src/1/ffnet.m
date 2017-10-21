%% Load data
load('Data_Problem1_regression.mat');
X = [X1 X2]';
T = (9*T1' + 4*T2' + 4*T3' + T4') / 18; %s0190440

i = randperm(size(X1, 1), 3000);
X_down = X(:, i);
T_down = T(i);

%% Generate performance data
hiddens = [1 2 3 4 5 10 15 20 25];
repeat_count = 10;
trainAlgs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm', 'trainbr'}; 
%trainAlgs = {'trainlm'}; 
transferFcns = {'logsig', 'tansig'};
data = {};

for hidden_count=hiddens
    for trainAlg=trainAlgs
        trainFc = char(trainAlg);
        for transferFcn=transferFcns
            transferFc = char(transferFcn);
            for j=1:repeat_count
                net = feedforwardnet(hidden_count, trainFc);
                net.trainParam.showWindow = false;
                net.divideParam.trainRatio = 1/3;
                net.divideParam.valRatio = 1/3;
                net.divideParam.testRatio = 1/3;
                net.layers{1}.transferFcn = transferFc;
                %output layer transferFcn always purelin for regression purposes
                
                tic;
                [net, tr] = train(net, X_down, T_down);
                time = toc;
                
                X_train = X_down(:, tr.trainInd);
                T_train = T_down(tr.trainInd);
                X_test = X_down(:, tr.testInd);
                T_test = T_down(tr.testInd);
                
                T_train_sim = sim(net, X_train);
                T_test_sim = sim(net, X_test);

                mserror_train = mean((T_train - T_train_sim).^2); %== tr.best_perf
                mserror_test = mean((T_test - T_test_sim).^2); %== tr.best_tperf
                
                data{end+1, 1} = hidden_count;
                data{end, 2} = trainFc;
                data{end, 3} = transferFc;
                data{end, 4} = time;
                data{end, 5} = mserror_train;
                data{end, 6} = mserror_test;
            end
        end
    end
end

tbl = cell2table(data, 'VariableNames', {'HiddenCount', 'trainFc', 'transferFc', 'Time', 'MSE_train', 'MSE_test'});

%% Analyze performance data
%stats = grpstats(tbl, {'HiddenCount', 'trainFc', 'transferFc'}, {'median'}, 'DataVars', {'Time', 'MSE_train', 'MSE_test'});
%group_plots(tbl, 'transferFc', size(transferFcns, 2), 20, 30)
group_plots(tbl, 'HiddenCount', size(hiddens, 2), 20, 30)

%% Plot test set and prediction
net = feedforwardnet(25, 'trainbr');
%net.trainParam.showWindow = false;
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 6;
net.divideParam.trainRatio = 1/3;
net.divideParam.valRatio = 1/3;
net.divideParam.testRatio = 1/3;
net.layers{1}.transferFcn = 'tansig';
%output layer transferFcn always purelin for regression purposes

tic;
[net, tr] = train(net, X_down, T_down);
time = toc;

X_train = X_down(:, tr.trainInd);
T_train = T_down(tr.trainInd);
X_test = X_down(:, tr.testInd);
T_test = T_down(tr.testInd);

T_train_sim = sim(net, X_train);
T_test_sim = sim(net, X_test);

mserror_train = mean((T_train - T_train_sim).^2); %== tr.best_perf
mserror_test = mean((T_test - T_test_sim).^2); %== tr.best_tperf

plot3(X_test(1,:), X_test(2,:), T_test, 'r.')
hold on;
plot3(X_test(1,:), X_test(2,:), T_test_sim, 'b.')
xlabel('X_1');
ylabel('X_2');

sizex = 30;
sizey = 20;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);
saveas(gcf, 'Regression_trainbr_TestSet', 'pdf');

%%
sizex = 30;
sizey = 20;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);
saveas(gcf, 'Regression_trainbr_Perf', 'pdf');