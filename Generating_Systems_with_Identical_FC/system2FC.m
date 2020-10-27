% This is a piece of code for generating A matrices that when used in a
% dyanamical system of the form x(t) = A x(t-1) + e(t) give rise to almost
% identical functional connectivity matrices but are themselves arbitrarily
% distinct. The results are reported in the supplementary material of the
% manuscript 
% 
% E. Nozari et. al., "Is the brain macroscopically linear? A
% system identification of resting state dynamics", 2020.
% 
% Copyright (C) 2020, Erfan Nozari
% All rights reserved.

%% Decision variables
n = 10;                                                                     % Dimension of A
sigma = 1;                                                                  % Standard deviation of noise
n_net = 10;                                                                 % Number of different A's to generate
N = 1e3;                                                                    % Length of time series to simulate
run_algorithm = 1;                                                          % Whether to run the algorithm (or load the results of previous runs, only for plotting)
plot_graphics = 1;                                                          % Whether to plot graphics

%% Book-keeping
A_rec = nan(n, n, n_net);                                                   % Record of A matrices
FC_rec = nan(n, n, n_net);                                                  % Record of functional connectivity (FC) matrices
spec_rec = nan(n, n_net);                                                   % Record of spectra (eigenvalues)
noise_rec = nan(n, N, n_net);                                               % Record of noise sequences used, for reproducibility

%% The main algorithm
if run_algorithm
    x0 = ones(n, 1);                                                        % The initial states
    options = optimoptions('fminunc', 'MaxFunctionEvaluations', 1e5, 'Display', 'off'); % fminunc options
    
    for i_net = 1:n_net
        A = randn(n);
        A = A / max(abs(eig(A))) * (1+rand)/2;                              % Stabilizing the generated A
        noise = sigma * randn(n, N);                                        % White noise sequence. The PSD does not matter though.
        if i_net > 1                                                        % The second to last A matrices are matched to the first. The first sets the target FC.
            [A, ~, exitflag] = fminunc(@(A)objfun(A, x0, noise, FC_rec(:, :, 1)), A, options);
            if exitflag <= 0
                warning(['Minimum not found for i_net = ' num2str(i_net)])
            end
        end
        x = ones(n, N);
        for t = 2:N                                                         % Simulating the system forward
            x(:, t) = A * x(:, t-1) + noise(:, t);
        end
        FC = corrcoef(x');

        A_rec(:, :, i_net) = A;
        FC_rec(:, :, i_net) = FC;
        spec_rec(:, i_net) = eig(A);
        noise_rec(:, :, i_net) = noise;
    end
    
    min_FC_corr = min(min(corrcoef(reshape(FC_rec, n^2, n_net))));          % Minimum (worst-case) FC similarity between any two pair of A matrices
    A_corr = corrcoef(reshape(A_rec, n^2, n_net));
    A_corr(logical(eye(n))) = nan;
    min_A_corr = min(min(abs(A_corr)));                                     % Minimum (best-case) A similarity between any pair of A matrices
    max_A_corr = max(max(abs(A_corr)));                                     % Maximum (worst-case) A similarity between any pair of A matrices
    save system2FC_data.mat n sigma n_net N A_rec FC_rec spec_rec noise_rec x0 options min_FC_corr ...
        min_A_corr max_A_corr A_corr
else
    load system2FC_data.mat A_rec FC_rec spec_rec
end

%% Graphics
if plot_graphics
    figure
    for i_net = 1:n_net
        subplot(5, 6, 3*(i_net-1)+1)
        imagesc(A_rec(:, :, i_net))
        colormap(gca, parula)
        axis equal
        axis([0.5 n+0.5 0.5 n+0.5])
        axis off
        if ismember(i_net, [1 2])
            title('$\mathbf{A}$', 'Interpreter', 'latex', 'FontSize', 15)
        end
        
        subplot(5, 6, 3*(i_net-1)+2)
        arrow3(-eye(2), eye(2), '2', 4, 6)
        hold on
        plot3(real(spec_rec(:, i_net)), imag(spec_rec(:, i_net)), ones(n, 1), 'x', 'linewidth', 3, ...
            'markersize', 10)
        view([0 90])
        axis([-1 1 -1 1 0 1])
        axis equal
        axis off
        if ismember(i_net, [1 2])
            title('Spectrum', 'Interpreter', 'latex', 'FontSize', 15)
        end
        
        subplot(5, 6, 3*(i_net-1)+3)
        FC = FC_rec(:, :, i_net);
        FC(logical(eye(n))) = 0;
        imagesc(FC)
        colormap(gca, parula)
        axis equal
        axis([0.5 n+0.5 0.5 n+0.5])
        axis off
        if ismember(i_net, [1 2])
            title('FC', 'Interpreter', 'latex', 'FontSize', 15)
        end
    end
    set(gcf, 'color', 'w', 'position', [1 1 975 800])
    export_fig system2FC.eps
end
        

%% Auxiliary functions
function J = objfun(A, x0, noise, FC0)                                      % This is the objective function called by fminunc above to compute the FC similarity between the to-be-tuned A and the target FC (FC0) given the fixed initial state x0 and noise sequence.
[n, N] = size(noise);
x = nan(n, N);
x(:, 1) = x0;
for t = 2:N
    x(:, t) = A * x(:, t-1) + noise(:, t);
end
FC = corrcoef(x');
R = corrcoef(FC(:), FC0(:));
J = -R(1, 2);
end