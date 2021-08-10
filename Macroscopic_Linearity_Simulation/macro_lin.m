function [R2_lin_rec, R2_nonlin_rec, h_opt_rec, x, y] = macro_lin(sweep, p, compute_R2, plot_graphics)
%MACRO_LIN Simulating the effects of spatial averaging, temporal averaging,
% observation noise, and dimensionality on nonlinear relationships between
% macroscopic brain signals, as implemented in the manuscript 
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2020.
% 
%   Input arguments
%
%   sweep: the property whose effect we want to study by sweeping over it.
%   Can take one of the values 'nave' (spatial averaging), 'fpass'
%   (temporal averaging/lowpass filtering), 'SNR' (signal to noise ratio),
%   or 'dim' (dimensionality).
% 
%   p: the exponent in the power law decay of spatial or temporal
%   autocorrelation of signals.
% 
%   compute_R2: binary flag indicating whether linear and nonlinear
%   cross-validated R^2 should be computed. Since will significantly add to
%   computation time, only set to 1 if needed.
% 
%   plot_graphics: whether to plot and save graphics. Set to 1 if calling
%   directly. Since this function is iterated over using
%   macro_lin_iterator.m, plot_graphics is set to 0 therein to prevent
%   multiple unwanted figures.
% 
%   Output arguments
% 
%   R2_lin_rec and R2_nonlin_rec: linear and nonlinear prediction R^2. Will
%   be set to NaN if compute_R2 = 0.
% 
%   h_opt_rec: only relevant if sweep = 'dim' and compute_R2 = 1, otherwise
%   it is set to NaN. In the former case, it gives the optimal window size
%   for the locally linear estimator used for nonlinear prediction by
%   sweeping over a range of h values.
% 
%   x and y: the two original sigmoidally-related signals that undergo
%   linearizing transformations according to sweep.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

if nargin < 1 || isempty(sweep)
    sweep = 'nave';
end
if nargin < 2 || isempty(p)
    p = 0;
end
if nargin < 3 || isempty(compute_R2)
    compute_R2 = 0;
end
if nargin < 4 || isempty(plot_graphics)
    plot_graphics = 0;
end

% Adding the parent directory and its sub-directories to the path
full_filename = mfilename('fullpath');
addpath(genpath(full_filename(1:strfind(full_filename, 'Macroscopic_Linearity_Simulation')-2)))

%% Variables applicable to all cases of sweep
nonlin_map = @tahh;
x_range = [-4.5 4.5];                                                       % Range of x. This is determined according to the nonlinearity of choice, which is tanh here.
if compute_R2
    switch sweep                    
        case {'nave', 'fpass', 'SNR'}                                       % For these cases, the nonlinear prediction can be obtained from the MMSE (optimal) prediction because the distributions are 2 dimensional
            MMSE_N_pdf = 200;
            MMSE_pdf_weight.method = 'normpdf';
            MMSE_memory = -8;
        case 'dim'                                                          % In this case, the dimension is swept over and MMSE becomes infeasible, thus using locally linear estimator
            LL_est_kernel = 'Gaussian';
    end
end

%% Sweeping over variable of choice and assessing its linearizing effects
switch sweep
    case 'nave'                                                             % Sweeping over the number of (x, y) pairs to average over, thus assessing the effect of averaging over space
        % Decision variables specific to sweep = 'nave'
        nave_vals = unique(round(logspace(0, 2, 9)));                       % The values of N_ave determining how many (x, y) pairs to average over
        n_nave = numel(nave_vals);
        N = 2e3;                                                            % Number of samples in x and y.
        uniform_rho = 1;                                                    % Binary flag determining whether the correlation coefficient between x pairs should be uniform (independent of distance) or not
        if ~uniform_rho
            rhomax = 0.5;                                                   % Maximum value of correlation coefficient between (spatially arbitrarily close) x pairs
            dmin = 1e-3;                                                    % The distance up to which rhomax applies
            rho = @(d)rhomax .* (d <= dmin) + rhomax*dmin^p./d.^p .* (d > dmin); % The function determining the relationship between correlation coefficient and distance
        end
        plot_sphere = 0;                                                    % Binary flag determining whether the sphere with x locations should be plotted
        
        % Sweeping over nave
        x_rec = nan(n_nave, N);
        y_rec = nan(n_nave, N);
        R2_lin_rec = nan(1, n_nave);
        R2_nonlin_rec = nan(1, n_nave);
        for i_nave = 1:n_nave
            nave = nave_vals(i_nave);
            if uniform_rho                                                  % Using spectral decomposition to obtain a uniform correlation coefficient of rho between all x pairs
                [V, ~] = qr([ones(nave, 1), randn(nave, nave-1)]);
                lambda = sqrt((1-p) / (1+p*(nave-1)));                      % This value of lambda creates a corrcoef of p between any pair of rows of X. This value is arbitrary and unimportant for nave = 1.
                Lambda = diag([1; lambda * ones(nave-1, 1)]);
                Q = V * Lambda * V';
            else
                xyz = rand(nave, 3) - 0.5;                                  % The (x, y, z) location of all x points in the unit sphere (i.e., sphere with unit diameter)
                xyz(sqrt(sum(xyz.^2, 2)) > 0.5, :) = [];                    % Only keeping the points that are inside the unit sphere
                while size(xyz, 1) < nave                                   % Keep adding (x, y, z) points until nave points inside the unit sphere are obtained
                    xyz_app = rand(nave - size(xyz, 1), 3) - 0.5;
                    xyz_app(sqrt(sum(xyz_app.^2, 2)) > 0.5, :) = [];
                    xyz = [xyz; xyz_app];
                end
                dist = squareform(pdist(xyz));
                cov_mat = rho(dist);                                        % Covariance matrix dependent on the distance
                cov_mat(logical(eye(nave))) = 1;
                [V, D] = eig(cov_mat);
                cov_mat = V * (max(0, D) / V);                              % Fixing the numerical erros that can cause small negative eigenvalues in cov_mat (which theoretically should not exist)
                Q = real(sqrtm(cov_mat));

                % Graphics
                if plot_graphics && plot_sphere
                    hf = figure;
                    arrow3(zeros(3), 0.8*eye(3), '2', 3, 5)
                    hold on
                    [sphere_x, sphere_y, sphere_z] = sphere(128);
                    hs = surfl(sphere_x/2, sphere_y/2, sphere_z/2); 
                    set(hs, 'FaceAlpha', 0.1)
                    shading interp
                    colormap('gray')
                    axis off
                    axis equal
                    plot3(xyz(:, 1), xyz(:, 2), xyz(:, 3), '.', 'markersize', 15, 'Color', matlab_red/3)
                    view([48 30])
                    hf.Color = 'w';
                end
            end
            X = Q * randn(nave, N);                                         % Creating signals with the prescribed (distance dependent or uniform) correlation coefficient
            X = (X - min(X(:))) / (max(X(:)) - min(X(:))) * (x_range(2) - x_range(1)) + x_range(1); % Linearly mapping to x_range
            Y = nonlin_map(X);
            x_rec(i_nave, :) = mean(X, 1);
            y_rec(i_nave, :) = mean(Y, 1);

            if compute_R2
                % Train-test separation
                N_test = round(0.1 * N);
                test_ind = randperm(N, N_test);
                train_ind = setdiff(1:N, test_ind);
                x_train = x_rec(i_nave, train_ind);
                x_test = x_rec(i_nave, test_ind);
                y_train = y_rec(i_nave, train_ind);
                y_test = y_rec(i_nave, test_ind);

                % Linear regression
                coeffs = polyfit(x_train, y_train, 1);
                y_test_hat = coeffs(1) * x_test + coeffs(2);
                R2_denom = sum((y_test - mean(y_test)).^2);
                R2_lin_rec(i_nave) = 1 - sum((y_test - y_test_hat).^2) / R2_denom;

                % Nonlinear regression
                MMSE_pdf_weight.rel_sigma = 0.02;
                y_test_hat = MMSE_est(x_train', y_train', x_test', MMSE_N_pdf, MMSE_pdf_weight, MMSE_memory)';
                R2_nonlin_rec(i_nave) = 1 - sum((y_test - y_test_hat).^2) / R2_denom;
            end
        end

        % Graphics
        if plot_graphics
            for i_nave = 1:n_nave
                nave = nave_vals(i_nave);
                hf = figure;
                hf.Position(4) = hf.Position(4) - 50;
                plot(x_rec(i_nave, :), y_rec(i_nave, :), '.', 'markersize', 10, 'color', matlab_purple)
                if nave == 1
                    xlabel('$x$ (a.u.)', 'Interpreter', 'latex')
                    ylabel('$y$ (a.u.)', 'Interpreter', 'latex')
                else
                    xlabel('$\langle x_i \rangle$ (a.u.)', 'Interpreter', 'latex')
                    ylabel('$\langle y_i \rangle$ (a.u.)', 'Interpreter', 'latex')
                end
                axis([x_range(1) x_range(2) -1.1 1.1])
                set(gca, 'fontsize', 35, 'xticklabel', [], 'yticklabel', [])
                grid on
                annotation(gcf, 'textbox', [0.2099 0.7238 0.2418 0.1071], 'LineStyle', 'none',...
                    'String', ['$N_{\rm ave} = ' num2str(nave) '$'], 'FontSize', 35, ...
                    'Interpreter', 'latex');
                hf.Color = 'w';
                export_fig(['macro_lin_nave_' num2str(nave) '.eps'])
            end
        end

    case 'fpass'                                                            % Sweeping over the cutoff frequency of a low-pass filter, thus assessing the effects of averaging over time
        % Decision variables specific to sweep = 'fpass'
        fpass_vals = logspace(-4, 0, 13);                                   % Values of normalized cutoff frequencies of lowpass filters applied to x and y.
        n_fpass = numel(fpass_vals);
        f_bandwidth_pre = min(10^(1/p-3), 1);                               % The bandwidth of x before being filtered. This is computed from the fact that the magnitude of the Fourier transform of x is given by min(1, 10^{-3p}/f^p), which equals to 0.1 at f = f_bandwidth_pre.                              
        N = 2e4 * floor(1 / f_bandwidth_pre);                               % We want to have 2e4 points if the bandwidth of x fills the entire [0, 1]. Otherwise, particularly if x is very lowpass, then a longer time series x is needed to capture the frequency content of x.
        
        % Generating the base signals (without lowpass filtering) according the value of p
        N_gen = 10 * N;                                                     % The number of data points used for generation of x. We first generate 10 times longer signal and then will cut its center portion to avoid edge effects of the naive filtering we are applying.
        fspan = linspace(0, 2, N_gen+1);                                    % Frequency span, in normalized frequency. Sampling frequency is 2 in normalized frequency.
        fspan(end) = [];
        fspan_half = fspan(2:N_gen/2+1);
        filter_half = min(1 ./ fspan_half.^p, (1e3)^p);                     % Capping the 1/f PSD at normalized f=1e-3, since otherwise it goes to infinity in theory.
        filter = [0, filter_half, fliplr(filter_half(1:end-1))];
        white_sig = randn(1, N_gen);                                        % White signal that will be filtered using filter to generate original (before lowpass filtering) x
        colored_sig = real(ifft(fft(white_sig) .* filter));
        colored_sig = colored_sig(fix(end*0.45)+(1:N));                      % Avoiding edge effects
        x = (colored_sig - min(colored_sig)) / (max(colored_sig) - min(colored_sig)) * (x_range(2) - x_range(1)) + x_range(1); % Linearly mapping colored_sig to x_range
        y = nonlin_map(x);
        
        % Sweeping over fpass
        R2_lin_rec = nan(1, n_fpass);
        R2_nonlin_rec = nan(1, n_fpass);
        x_rec = cell(n_fpass, 1);
        y_rec = cell(n_fpass, 1);
        for i_fpass = 1:n_fpass
            fpass = fpass_vals(i_fpass);
            if p == 0 && fpass < 0.03 || p <= 0.25 && fpass < 0.02 || p <= 0.5 && fpass < 3e-4
                continue
            end

            if fpass == 1
                x_filtered = x;
                y_filtered = y;
            else
                sigma = 1/2/pi/fpass;
                tspan = [fliplr(0:-0.5:-5*sigma), 0.5:0.5:5*sigma];
                h_filter = 1/2 * 1/sqrt(2*pi)/sigma * exp(-tspan.^2/2/sigma^2);
                x_filtered = conv(x, h_filter, 'same');
                y_filtered = conv(y, h_filter, 'same');
            end
            f_bandwidth_post = min(10*fpass, f_bandwidth_pre);
            ss_ratio = floor(1 / f_bandwidth_post);
            x_rec{i_fpass} = x_filtered(1:ss_ratio:end);
            y_rec{i_fpass} = y_filtered(1:ss_ratio:end);
            N = numel(x_rec{i_fpass});

            if compute_R2
                % Train-test separation            
                N_test = round(0.1 * N);
                test_ind = randperm(N, N_test);
                train_ind = setdiff(1:N, test_ind);
                x_train = x_rec{i_fpass}(train_ind);
                x_test = x_rec{i_fpass}(test_ind);
                y_train = y_rec{i_fpass}(train_ind);
                y_test = y_rec{i_fpass}(test_ind);

                % Linear regression
                coeffs = polyfit(x_train, y_train, 1);
                y_test_hat = coeffs(1) * x_test + coeffs(2);
                R2_denom = sum((y_test - mean(y_test)).^2);
                R2_lin_rec(i_fpass) = 1 - sum((y_test - y_test_hat).^2) / R2_denom;

                % Nonlinear regression
                MMSE_pdf_weight.rel_sigma = 0.02;
                y_test_hat = MMSE_est(x_train', y_train', x_test', MMSE_N_pdf, MMSE_pdf_weight, MMSE_memory)';
                R2_nonlin_rec(i_fpass) = 1 - sum((y_test - y_test_hat).^2) / R2_denom;
            end
        end

        % Graphics
        if plot_graphics
            for i_fpass = 1:n_fpass
                hf = figure;
                hf.Position(4) = hf.Position(4) - 50;
                plot(x_rec{i_fpass}, y_rec{i_fpass}, '.', 'markersize', 10, 'color', matlab_red)
                axis([x_range(1) x_range(2) -1.1 1.1])
                set(gca, 'fontsize', 35, 'xticklabel', [], 'yticklabel', [])
                grid on
                fpass = fpass_vals(i_fpass);
                if fpass == 1
                    xlabel('$x$ (a.u.)', 'Interpreter', 'latex')
                    ylabel('$y$ (a.u.)', 'Interpreter', 'latex')
                else
                    xlabel('LPF\{$x$\} (a.u.)', 'Interpreter', 'latex')
                    ylabel('LPF\{$y$\} (a.u.)', 'Interpreter', 'latex')
                end
                annotation(gcf, 'textbox', [0.15 0.7238 0.2418 0.1071], 'LineStyle', 'none', ...
                    'String', ['$f_{\rm cutoff} = ' num2str(fpass_vals(i_fpass)) '$'], 'FontSize', 35, ...
                    'Interpreter', 'latex');
                set(gcf, 'Color', 'w')
                export_fig(['macro_lin_fpass_' num2str(i_fpass) '.eps'])
            end
        end

    case 'SNR'                                                              % Sweeping over observation signal to noise ratio, thus assessing the effect of observation noise
        % Decision variables specific to sweep = 'SNR'
        SNR_vals = logspace(0, 2, 9);
        n_SNR = numel(SNR_vals);
        N = 2e3;                                                            % Number of samples in x and y. Note that this is later overwritten if sweep = 'fpass' or 'dim'.
        
        % Generating the base signals (without noise)
        white_sig = randn(1, N);
        x = (white_sig - min(white_sig)) / (max(white_sig) - min(white_sig)) * (x_range(2) - x_range(1)) + x_range(1);
        y = nonlin_map(x);

        % Sweeping over SNR
        R2_lin_rec = nan(1, n_SNR);
        R2_nonlin_rec = nan(1, n_SNR);
        x_rec = nan(n_SNR, N);
        y_rec = nan(n_SNR, N);
        for i_SNR = 1:n_SNR
            SNR = SNR_vals(i_SNR);
            x_rec(i_SNR, :) = x + std(x) / SNR * randn(size(x));
            y_rec(i_SNR, :) = y + std(y) / SNR * randn(size(y));

            if compute_R2
                % Train-test separation
                N_test = round(0.1 * N);
                test_ind = randperm(N, N_test);
                train_ind = setdiff(1:N, test_ind);
                x_train = x_rec(i_SNR, train_ind);
                x_test = x_rec(i_SNR, test_ind);
                y_train = y_rec(i_SNR, train_ind);
                y_test = y_rec(i_SNR, test_ind);

                % Linear regression
                coeffs = polyfit(x_train, y_train, 1);
                y_test_hat = coeffs(1) * x_test + coeffs(2);
                R2_denom = sum((y_test - mean(y_test)).^2);
                R2_lin_rec(i_SNR) = 1 - sum((y_test - y_test_hat).^2) / R2_denom;

                % Nonlinear regression
                MMSE_pdf_weight.rel_sigma = 0.02 + 0.02 / SNR;                   % This is set heuristically by trial and error
                y_test_hat = MMSE_est(x_train', y_train', x_test', MMSE_N_pdf, MMSE_pdf_weight, MMSE_memory)';
                R2_nonlin_rec(i_SNR) = 1 - sum((y_test - y_test_hat).^2) / R2_denom;
            end
        end

        % Graphics
        if plot_graphics
            for i_SNR = 1:n_SNR
                hf = figure;
                hf.Position(4) = hf.Position(4) - 50;
                plot(x_rec(i_SNR, :), y_rec(i_SNR, :), '.', 'markersize', 10, 'color', matlab_colors(7))
                xlabel('$x$ + noise (a.u.)', 'Interpreter', 'latex')
                ylabel('$y$ + noise (a.u.)', 'Interpreter', 'latex')
                xlims = get(gca, 'xlim');
                xlim(max(abs(xlims)) * [-1 1]);
                ylims = get(gca, 'ylim');
                ylim(max(abs(ylims)) * [-1 1]);
                set(gca, 'fontsize', 35, 'xticklabel', [], 'yticklabel', [])
                grid on
                annotation(gcf, 'textbox', [0.1616 0.7832 0.3418 0.1071], 'LineStyle', 'none',...
                'String', ['SNR = $' num2str(SNR_vals(i_SNR)) '$'], 'FontSize', 35, 'Interpreter', 'latex');
                hf.Color = 'w';
                export_fig(['macro_lin_snr_' num2str(SNR_vals(i_SNR)) '.eps'])
            end
        end

    case 'dim'                                                              % Sweeping over the dimension of the nonlinear relationship, thus assessing the effect of limited data points
        % Decision variables specific to sweep = 'dim'
        dim_vals = unique(round(logspace(0, 2, 11)));
        n_dim = numel(dim_vals);
        N = 1e3;
        n_cv = 5;
        LL_est_h = logspace(-1, 1, 21);
        
        % Sweeping over dim
        R2_lin_rec = nan(1, n_dim);
        R2_nonlin_rec = nan(1, n_dim);
        n_h = numel(LL_est_h);
        if n_h > 1
            h_opt_rec = nan(1, n_dim);
        end
        for i_dim = 1:n_dim
            dim = dim_vals(i_dim);
            X = randn(dim, N);
            X = X ./ max(sqrt(sum(X.^2, 1)));
            sum_X = sum(X, 1);
            x = (x_range(2) - x_range(1)) / (max(sum_X) - min(sum_X)) * (sum_X - min(sum_X)) + x_range(1);
            y = nonlin_map(x);

            if dim == 1
                X1 = X;
                y1 = y;
            elseif dim == 2
                X2 = X;
                y2 = y;
            elseif dim == 3
                X3 = X;
                y3 = y;
            end

            if compute_R2
                % Train-test separation
                if n_h > 1
                    test_ind = randperm(N, round(N/10));
                    train_ind = setdiff(1:N, test_ind);
                    X_train = X(:, train_ind);
                    X_test = X(:, test_ind);
                    y_train = y(train_ind);
                    y_test = y(test_ind);
                    R2_rec = nan(n_h, 1);
                    for i_h = 1:n_h
                        y_test_hat = LL_est(X_train', y_train', X_test', LL_est_kernel, LL_est_h(i_h))';
                        R2_rec(i_h) = 1 - sum((y_test_hat - y_test).^2) / sum((y_test - mean(y_test)).^2);
                    end
                    [~, max_ind] = max(R2_rec);
                    h = LL_est_h(max_ind);
                    h_opt_rec(i_dim) = h;
                else
                    h = LL_est_h;
                end

                y_test_hat_lin = nan(1, N);
                y_test_hat_nonlin = nan(1, N);
                rand_ind = randperm(N);
                break_pts = round(linspace(0, N, n_cv+1));
                for i_cv = 1:n_cv
                    test_ind = rand_ind(break_pts(i_cv)+1:break_pts(i_cv+1));
                    train_ind = setdiff(1:N, test_ind);
                    X_train = X(:, train_ind);
                    N_train = numel(train_ind);
                    X_test = X(:, test_ind);
                    N_test = numel(test_ind);
                    y_train = y(train_ind);
                    y_test = y(test_ind);

                    % Linear regression
                    coeffs = lsqminnorm([X_train', ones(N_train, 1)], y_train')';
                    y_test_hat_lin(test_ind) = coeffs * [X_test; ones(1, N_test)];

                    % Nonlinear regression                
                    y_test_hat_nonlin(test_ind) = LL_est(X_train', y_train', X_test', LL_est_kernel, h)';
                end
                R2_denom = sum((y_test - mean(y_test)).^2);
                R2_lin_rec(i_dim) = 1 - sum((y_test - y_test_hat_lin).^2) / R2_denom;
                R2_nonlin_rec(i_dim) = 1 - sum((y_test - y_test_hat_nonlin).^2) / R2_denom;
            end
        end

        % Graphics
        if plot_graphics
            hf = figure;
            hf.Position(4) = hf.Position(4) - 50;
            scatter(X1, y1, 10, y1, 'filled')
            xlabel('$x$ (a.u.)', 'Interpreter', 'latex')
            ylabel('$y$ (a.u.)', 'Interpreter', 'latex')
            xlim(max(abs(X1)) * [-1.1 1.1])
            ylim([-1.1 1.1])
            set(gca, 'fontsize', 35, 'xticklabel', [], 'yticklabel', [])
            grid on
            colormap copper
            hf.Color = 'w';
            export_fig macro_lin_dim_1.eps

            hf = figure;
            scatter3(X2(1, :), X2(2, :), y2, 10, y2, 'filled')
            hx = xlabel('$x_1$ (a.u.)', 'Interpreter', 'latex');
            hy = ylabel('$x_2$ (a.u.)', 'Interpreter', 'latex');
            zlabel('$y$ (a.u.)', 'Interpreter', 'latex')
            xlims = get(gca, 'xlim');
            xlim(min(abs(xlims)) * 0.7 * [-1 1]);
            ylims = get(gca, 'ylim');
            ylim(min(abs(ylims)) * 0.7 * [-1 1]);
            set(gca, 'fontsize', 35, 'xticklabel', [], 'yticklabel', [], 'ZTickLabel', [])
            grid on
            view([22 9])
            colormap copper
            drawnow
            hy.Position = [0.5 -0.1 -1.1];
            hx.Position(3) = hx.Position(3) + 0.35;
            hx.Position(1) = hx.Position(1) + 0.1;
            hf.Color = 'w';
            export_fig macro_lin_dim_2.eps

            hf = figure;
            scatter3(X3(1, :), X3(2, :), X3(3, :), 10, y3, 'filled')
            hx = xlabel('$x_1$ (a.u.)', 'Interpreter', 'latex');
            hy = ylabel('$x_2$ (a.u.)', 'Interpreter', 'latex');
            zlabel('$x_3$ (a.u.)', 'Interpreter', 'latex')
            set(gca, 'fontsize', 35, 'xticklabel', [], 'yticklabel', [], 'zticklabel', [])
            xlims = get(gca, 'xlim');
            xlim(min(abs(xlims)) * 0.7 * [-1 1]);
            ylims = get(gca, 'ylim');
            ylim(min(abs(ylims)) * 0.7 * [-1 1]);
            zlims = get(gca, 'zlim');
            zlim(min(abs(zlims)) * 0.7 * [-1 1]);
            grid on
            view([38 10])
            colormap copper
            hc = colorbar('Ticks', []);
            hc.Position(4) = hc.Position(4) - 0.16;
            hc.Position(2) = hc.Position(2) + 0.09;
            hc.Position(1) = hc.Position(1) + 0.02;
            hc.Title.String = '$y$ (a.u.)';
            hc.Title.Interpreter = 'latex';
            hc.Title.Rotation = 90;
            hc.Title.Position = [60 134 0];
            drawnow
            ha = gca;
            ha.Position(3) = ha.Position(3) - 0.1;
            drawnow
            hx.Position(2) = hx.Position(2) + 0.1;
            hx.Position(3) = hx.Position(3) + 0.25;
            hx.Position(1) = hx.Position(1) + 0.1;
            hy.Position(3) = hy.Position(3) + 0.3;
            hy.Position(2) = hy.Position(2) - 0.45;
            hf.Color = 'w';
            export_fig macro_lin_dim_3.eps
        end
end

if ~exist('h_opt_rec', 'var')
    h_opt_rec = [];
end
end