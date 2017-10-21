%% Load data
A = prprob();
A(A == 0) = -1;

A = [A; A];
%A = [A A];

num_letters = size(A, 2);
%% Display all
h = 14;
result = zeros(h, 5*num_letters);
for i=1:num_letters
    offset = 5*(i-1);
    result(:, (1:5) + offset) = reshape(A(:, i), 5, [])';
end
imshow(result);

%% Test distortion
letter = A(:, 1);
dletter = distort(letter, 3);
img = vec2img(dletter);
colormap gray;
imagesc(img);

%% Train Hopfield
P = 33;
T = A(:, 1:P);
net = newhop(T);

%% Test exact input
[Y,~,~] = sim(net, P, [], T);
for i = 1:P
    letter = Y(:,i);
    img = vec2img(letter);
    subplot(1, P, i);
    imshow(img)
end

%% Hopfield - distorted
num_distortions = 3;
nums_iterations = [1 2 5 10 20 50 100 200 500];
Ps = 1:num_letters;
repeat = 10;
error = zeros(numel(nums_iterations), num_letters);
for P=Ps
    for i=1:numel(nums_iterations)
        tmp = zeros(1, repeat);
        for j=1:repeat
            tmp(j) = hop_perf(A, P, nums_iterations(i), num_distortions);
        end
        error(i, P) = median(tmp);
        if error(i, P) == 0
           error(i+1:end, P) = NaN;
           break;
        end
    end
end

%     colormap gray;
%     for i = 1:P
%         subplot(P, 2, 2*i-1);
%         imagesc(vec2img(Y(:,i)))
%         axis off;
%         subplot(P, 2, 2*i);
%         imagesc(vec2img(Yr(:,i)))
%         axis off;
%     end

%%
subplot(3,1,1)
range = 1:3;
plot(Ps, error(range, :)', 'o-');
xlim([0 num_letters]);
ylim([0 110]);
xlabel('P');
ylabel('Errors');
h = legend(num2str((nums_iterations(range)).'), 'Location', 'northwest');
set(get(h,'title'),'string','Iterations');

subplot(3,1,2)
range = 4:6;
plot(Ps, error(range, :)', 'o-');
xlim([0 num_letters]);
ylim([0 110]);
xlabel('P');
ylabel('Errors');
h = legend(num2str((nums_iterations(range)).'), 'Location', 'northwest');
set(get(h,'title'),'string','Iterations');

subplot(3,1,3)
range = 7:9;
plot(Ps, error(range, :)', 'o-');
xlim([0 num_letters]);
ylim([0 110]);
xlabel('P');
ylabel('Errors');
h = legend(num2str((nums_iterations(range)).'), 'Location', 'northwest');
set(get(h,'title'),'string','Iterations');

sizex = 10;
sizey = 15;
set(gcf, 'PaperPosition', [0 0 sizex sizey]);
set(gcf, 'PaperSize', [sizex sizey]);
saveas(gcf, 'Letters_DoublePerf', 'pdf');