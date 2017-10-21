%% Load data
data = readtable('winequality_data\winequality-white.csv');
c1 = data(data.quality == 5, :);
c2 = data(data.quality == 6, :);
c = [c1; c2];
c.label1 = c.quality == 5;
c.label2 = c.quality == 6;
input = table2array(c);
X = input(:, 1:end-3)';
y = input(:, end-1:end)';

%% Plot PCA quality
% [~,d] = eig(cov(X'));
% d_sorted = flipud(diag(d));
% d_cumquality = cumsum(d_sorted) / sum(d_sorted);
% plot(1-d_cumquality, '*-');
% xlabel('q')
% ylabel('quality loss')
% 
% sizex = 15;
% sizey = 10;
% set(gcf, 'PaperPosition', [0 0 sizex sizey]);
% set(gcf, 'PaperSize', [sizex sizey]);
% saveas(gcf, 'WineQualityLoss', 'pdf');

%% Perform PCA and calculate errors
[X_norm2, mapsettings] = mapstd(X);
% q maxfrac
% 1 0.15
% 2 0.12
% 3 0.10
maxfrac = 0.10;
[E2, pcasettings] = processpca(X_norm2, maxfrac);
size(E2, 1)
X_norm_hat2 = processpca('reverse', E2, pcasettings);
X_hat2 = mapstd('reverse', X_norm_hat2, mapsettings);

rmse2_norm = sqrt(mean((X_norm2-X_norm_hat2).^2, 2));
rmse2 = sqrt(mean((X-X_hat2).^2, 2));
disp([rmse2_norm rmse2]')

%% Plot 3D samples and predictions
% ax1 = subplot(121);
% E21 = E2(y(1,:) == 1);
% E22 = E2(y(2,:) == 1);
% index1 = y(1,:) == 1;
% index2 = y(2,:) == 1;
% plot3(E2(1,index1), E2(2,index1), E2(3,index1), 'r.');
% hold on;
% plot3(E2(1,index2), E2(2,index2), E2(3,index2), 'b.');
% title('Training data');
% xlabel('C_1');
% ylabel('C_2');
% zlabel('C_3');
% legend('Quality 5', 'Quality 6', 'Location','southoutside');
% xlim([-4 4]);
% ylim([-4 4]);
% zlim([-4 4]);
% 
% ax2 = subplot(122);
% net = patternnet();
% %net.trainParam.max_fail = 1000;
% [net, tr] = train(net, E2, y);
% range = -4:0.5:4;
% [U,V,W] = meshgrid(range,range,range);
% input = [U(:) V(:) W(:)]';
% T_sim = sim(net, input);
% index1 = T_sim(1,:) >= 0.5;
% index2 = T_sim(1,:) < 0.5;
% plot3(input(1,index1), input(2,index1), input(3,index1), 'r.');
% hold on;
% plot3(input(1,index2), input(2,index2), input(3,index2), 'b.');
% title('Predictions');
% xlabel('C_1');
% ylabel('C_2');
% zlabel('C_3');
% legend('Quality 5', 'Quality 6', 'Location','southoutside');
% 
% Link = linkprop([ax1, ax2], {'CameraUpVector', 'CameraPosition', 'CameraTarget'});
% setappdata(gcf, 'StoreTheLink', Link);
% 
% sizex = 30;
% sizey = 20;
% set(gcf, 'PaperPosition', [0 0 sizex sizey]);
% set(gcf, 'PaperSize', [sizex sizey]);
% saveas(gcf, 'WinePrediction', 'pdf');
% return

%% Compare network architectures and training functions
hiddens = [1 2 3 4 5 10 15 20 25];
trainAlgs = {'trainscg', 'trainrp', 'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg', 'trainlm'}; %, 'trainbr'
repeat_count = 10;
data = {};

for hidden_count=hiddens
    for trainAlg=trainAlgs
        trainFc = char(trainAlg);
        for j=1:repeat_count
            net = patternnet(hidden_count, trainFc);
            net.trainParam.showWindow = false;

            tic;
            [net, tr] = train(net, E2, y);
            time = toc;

            X_val = E2(:, tr.valInd);
            T_val = y(:, tr.valInd);
            T_val_sim = sim(net, X_val);
            [c,~,~,~] = confusion(T_val, T_val_sim);

            data{end+1, 1} = hidden_count;
            data{end, 2} = trainFc;
            data{end, 3} = time;
            data{end, 4} = 100*(1-c); %CCR
        end
    end
end

tbl = cell2table(data, 'VariableNames', {'HiddenCount', 'trainFc', 'Time', 'CCR'});

%% Plot training function differences
stats = grpstats(tbl, {'trainFc'}, {'median'}, 'DataVars', {'Time', 'CCR'});

subplot(121);
bar(stats.median_CCR');
set(gca,'xticklabel', stats.trainFc);
xtickangle(90);
xlabel('trainFc');
ylabel('CCR (%)');
ylim([0 100]);

subplot(122);
bar(stats.median_Time);
set(gca,'xticklabel', stats.trainFc);
xtickangle(90);
xlabel('trainFc');
ylabel('Time (s)');

sizex = 15;
sizey = 10;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);
saveas(gcf, 'WineResultsPCA', 'pdf');

%% Plot network architecture differences
stats = grpstats(tbl, {'HiddenCount'}, {'median'}, 'DataVars', {'Time', 'CCR'});

subplot(121);
bar(stats.median_CCR');
set(gca,'xticklabel', stats.HiddenCount);
xlabel('HiddenCount');
ylabel('CCR (%)');
ylim([0 100]);

subplot(122);
bar(stats.median_Time);
set(gca,'xticklabel', stats.HiddenCount);
xlabel('HiddenCount');
ylabel('Time (s)');

sizex = 15;
sizey = 10;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);
saveas(gcf, 'WineResultsArchPCA', 'pdf');