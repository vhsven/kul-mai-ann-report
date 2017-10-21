function error = hop_perf(A, P, num_iterations, num_distortions)
    T = A(:, 1:P);
    net = newhop(T);
    Tn = distortall(T, num_distortions);
    [Y,~,~] = sim(net, [P, num_iterations], [], Tn);
    Y = Y{1,end};
    Yr = zeros(size(Y));
    Yr(Y >= 0) = 1;
    Yr(Y < 0) = -1;

    errors = sum(Yr ~= T);
    error = sum(errors);
end