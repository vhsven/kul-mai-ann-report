load('Data_Problem1_regression.mat');
X = [X1 X2]';
T = (9*T1' + 4*T2' + 4*T3' + T4') / 18; %s0190440

i = randperm(size(X1, 1), 3000);
X_down = X(:, i);
T_down = T(i);

net = feedforwardnet(10, 'trainlm');
%net.trainParam.showWindow = false;
net.divideParam.trainRatio = 1/3;
net.divideParam.valRatio = 1/3;
net.divideParam.testRatio = 1/3;
[net, tr] = train(net, X_down, T_down);

X_train = X_down(:, tr.trainInd);
T_train = T_down(tr.trainInd);
X_test = X_down(:, tr.testInd);
T_test = T_down(tr.testInd);

figure;
plot3(X_train(1,:), X_train(2,:), T_train, '.');
xlabel('X_1');
ylabel('X_2');
zlabel('f(X_1, X_2)');
title('Training samples');
sizex = 15;
sizey = 10;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);
saveas(gcf, 'TrainPoints', 'pdf');

T_train_sim = sim(net, X_train);
T_test_sim = sim(net, X_test);

figure;
subplot(121);
plot3(X_test(1,:), X_test(2,:), T_test, '.');
subplot(122);
plot3(X_test(1,:), X_test(2,:), T_test_sim, '.');

%figure;
%[m,b,r] = postreg(T_test_sim, T_test);
mserror_train = mean((T_train - T_train_sim).^2) %== tr.best_perf
mserror_test = mean((T_test - T_test_sim).^2) %== tr.best_tperf