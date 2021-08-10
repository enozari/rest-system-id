function mlI_out = macro_lin_Izhikevic(sweep, SNR, run_algorithm, compute_R2, plot_graphics)
%MACRO_LIN_IZHIKEVIC Simulating the effects of spatial and temporal
%averaging on the nonlinear dynamics of the Izhikevic spiking model, as
%implemented in the manuscript
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2021.
% 
%   Input arguments
%
%   sweep: the property whose effect we want to study by sweeping over it.
%   Can take one of the values 'nave' (spatial averaging) or 'fpass'
%   (temporal averaging/lowpass filtering).
% 
%   SNR: process (not observation) signal to noise ratio. No observation
%   noise is added here.
% 
%   compute_R2: binary flag indicating whether linear and nonlinear
%   cross-validated R^2 should be computed. Since will significantly add to
%   computation time, only set to true if needed.
% 
%   plot_graphics: whether to plot and save graphics. Set to true if
%   calling directly. Since this function is iterated over using
%   macro_lin_Izhikevic_iterator.m, plot_graphics is set to 0 therein to
%   prevent multiple unwanted figures.
% 
%   Output arguments:
% 
%   mlI_out: struct containing various data generated during the
%   simulations.
% 
%   Copyright (C) 2021, Erfan Nozari
%   All rights reserved.

if nargin < 1 || isempty(sweep)
    sweep = 'fpass';
end
if nargin < 2 || isempty(SNR)
    SNR = inf;
end
if nargin < 3 || isempty(run_algorithm)
    run_algorithm = 1;
end
if nargin < 4 || isempty(compute_R2)
    compute_R2 = 1;
end
if nargin < 5 || isempty(plot_graphics)
    plot_graphics = 1;
end

if run_algorithm
    rng('shuffle')
    rng_settings = rng;

    a = 0.02;
    b = 0.2;
    c = -65;
    d = 2;
    I = 7;
    
    T = 2000;
    u = nan(1, T);
    u(1) = -5.5;
    v = nan(1, T);
    v(1) = c;
    for t = 2:T
        if v(t-1) >= 30
            v(t) = c;
            u(t) = u(t-1) + d;
        else
            v(t) = v(t-1) + 0.1 * (0.04*v(t-1)^2 + 5*v(t-1) + 140 - u(t-1) + I);
            u(t) = u(t-1) + 0.1 * (a*b*v(t-1) - a*u(t-1));
        end
    end
    [~, peak_ind] = findpeaks(v);
    v_period = v(peak_ind(2)+1:peak_ind(3));
    u_period = u(peak_ind(2)+1:peak_ind(3));
    n_period = numel(v_period);

    sigma = std(v) / SNR;

    switch sweep
        case 'nave'
            nave_vals = logspace(0, 4, 5);                       % The values of N_ave determining how many (x, y) pairs to average over
            n_nave = numel(nave_vals);
    
            T = 20000;

            [v_bar_rec, u_bar_rec, v_bar_diff_rec] = deal(cell(n_nave, 1));
            R2_lin_rec = nan(1, n_nave);
            h_vals = logspace(-4, 2, 19);
            n_h = numel(h_vals);
            R2_nonlin_rec = nan(n_h, n_nave);
            for i_nave = 1:n_nave
                nave = nave_vals(i_nave);

                init_ind = randi(n_period, nave, 1);
                V0 = v_period(init_ind);
                U0 = u_period(init_ind);
                [V, U] = run_Izhikevic(a, b, c, d, I, sigma, T, nave, V0, U0);
 
                v_bar = mean(V, 1);
                v_bar_diff = diff(v_bar);
                v_bar = v_bar(1:end-1);
                u_bar = mean(U(:, 1:end-1), 1);

                v_bar = v_bar - mean(v_bar);
                u_bar = u_bar - mean(u_bar);

                v_bar_rec{i_nave} = v_bar;
                u_bar_rec{i_nave} = u_bar;
                v_bar_diff_rec{i_nave} = v_bar_diff;

                if compute_R2
                    [R2_lin_rec(i_nave), R2_nonlin_rec(:, i_nave)] = ...
                        R2_gen(v_bar, u_bar, v_bar_diff, h_vals);
                end
            end
            
            mlI_out.nave_vals = nave_vals;
    
        case 'fpass'
            fpass_vals = logspace(-4, 0, 5);
            n_fpass = numel(fpass_vals);
    
            [v_bar_rec, u_bar_rec, v_bar_diff_rec] = deal(cell(n_fpass, 1));
            R2_lin_rec = nan(1, n_fpass);
            h_vals = logspace(-4, 2, 19);
            n_h = numel(h_vals);
            R2_nonlin_rec = nan(n_h, n_fpass);
            for i_fpass = 1:n_fpass
                fpass = fpass_vals(i_fpass);
                nu = 1/2/pi/fpass;
                tspan_filter = [fliplr(0:-0.5:-5*nu), 0.5:0.5:5*nu];
                h_filter = 1/2 * 1/sqrt(2*pi)/nu * exp(-tspan_filter.^2/2/nu^2);
                h_filter = h_filter / sum(h_filter);
                
                T = max(1e4, 2*fix(numel(tspan_filter)));

                init_ind = randi(n_period);
                V0 = v_period(init_ind);
                U0 = u_period(init_ind);
                [v_bar, u_bar] = run_Izhikevic(a, b, c, d, I, sigma, 10*T, 1, V0, U0);

                v_bar = conv(v_bar, h_filter, 'same');
                v_bar = v_bar(fix(end*0.45)+(1:T+1));
                v_bar_diff = diff(v_bar);
                v_bar = v_bar(1:end-1);
                u_bar = conv(u_bar, h_filter, 'same');
                u_bar = u_bar(fix(end*0.45)+(1:T));

                v_bar = v_bar - mean(v_bar);
                u_bar = u_bar - mean(u_bar);

                v_bar_rec{i_fpass} = v_bar;
                u_bar_rec{i_fpass} = u_bar;
                v_bar_diff_rec{i_fpass} = v_bar_diff;

                if compute_R2
                    [R2_lin_rec(i_fpass), R2_nonlin_rec(:, i_fpass)] = ...
                        R2_gen(v_bar, u_bar, v_bar_diff, h_vals);
                end
            end
            
            mlI_out.fpass_vals = fpass_vals;
    end
    
    mlI_out.rng_settings = rng_settings;
    mlI_out.a = a;
    mlI_out.b = b;
    mlI_out.c = c;
    mlI_out.d = d;
    mlI_out.I = I;
    mlI_out.T = T;
    mlI_out.v_period = v_period;
    mlI_out.u_period = u_period;
    mlI_out.SNR = SNR;
    mlI_out.v_bar_rec = v_bar_rec;
    mlI_out.u_bar_rec = u_bar_rec; 
    mlI_out.v_bar_diff_rec = v_bar_diff_rec;
    mlI_out.R2_lin_rec = R2_lin_rec;
    mlI_out.R2_nonlin_rec = R2_nonlin_rec;
    mlI_out.h_vals = h_vals;
    
    save mlI_data.mat mlI_out
else
    load mlI_data_SNRinf.mat mlI_out
end

% Graphics
if plot_graphics
    switch sweep
        case 'nave'
            n_par = numel(mlI_out.nave_vals);
            par_vals = mlI_out.nave_vals;
            color = matlab_purple;
            x_label = '$\langle v_i \rangle$';
            y_label = '$\langle u_i \rangle$';
            z_label = '$\langle dv_i/dt \rangle$';
            annotation_string = '$N_{\rm ave} = ';
            eps_string = 'mlI_nave_';
        case 'fpass'
            n_par = numel(mlI_out.fpass_vals);
            par_vals = mlI_out.fpass_vals;
            color = matlab_red;
            x_label = 'LPF$\{v\}$';
            y_label = 'LPF$\{u\}$';
            z_label = '$d/dt$ LPF$\{v\}$';
            annotation_string = '$f_{\rm cutoff} = ';
            eps_string = 'mlI_fpass_';
    end
    
    for i_par = 1:n_par
        par = par_vals(i_par);
        figure
        plot3(mlI_out.v_bar_rec{i_par}, mlI_out.u_bar_rec{i_par}, ...
            mlI_out.v_bar_diff_rec{i_par}, '.', 'markersize', 10, 'color', color)
        if par == 1
            hx = xlabel('$v$', 'Interpreter', 'latex');
            hy = ylabel('$u$', 'Interpreter', 'latex');
            zlabel('$dv/dt$', 'Interpreter', 'latex')
        else
            hx = xlabel(x_label, 'Interpreter', 'latex');
            hy = ylabel(y_label, 'Interpreter', 'latex');
            zlabel(z_label, 'Interpreter', 'latex')
        end
        ha = gca;
        set(ha, 'fontsize', 20, 'ticklabelinterpreter', 'latex')
        hx.Position = [mean(ha.XLim) 1.3*ha.YLim(1)-0.3*ha.YLim(2) ha.ZLim(1)];
        hy.Position = [1.3*ha.XLim(1)-0.3*ha.XLim(2) mean(ha.YLim)  ha.ZLim(1)];
        grid on
        annotation(gcf, 'textbox', [0.163471428571429 0.816657142857144 0.2418 0.1071], ...
            'LineStyle', 'none', 'Interpreter', 'latex', ...
            'String', [annotation_string num2str(par) '$'], 'FontSize', 35);
        exportgraphics(gca, [eps_string num2str(par) '_SNRinf.eps'])
    end
    
    if compute_R2
        figure
        [R2_nonlin_rec_max, max_ind] = max(mlI_out.R2_nonlin_rec, [], 1);
        h_opt_rec = mlI_out.h_vals(max_ind);
        semilogx(mlI_out.nave_vals, mlI_out.R2_lin_rec, '.-', ...
            mlI_out.nave_vals, R2_nonlin_rec_max, '.-', 'linewidth', 5, 'markersize', 20)
        xlabel('nave')
        ylabel('R2')

        figure
        loglog(mlI_out.nave_vals, h_opt_rec, '.-', 'linewidth', 5, 'markersize', 20)
        xlabel('nave')
        ylabel('h__opt')

        figure
        [Nave_vals, H_vals] = meshgrid(mlI_out.nave_vals, mlI_out.h_vals);
        mesh(Nave_vals, H_vals, mlI_out.R2_nonlin_rec)
        set(gca, 'xscale', 'log', 'yscale', 'log')
        xlabel('nave')
        ylabel('h')
        zlabel('R2__nonlin')
    end
end

%% Auxiliary functions
function [V, U] = run_Izhikevic(a, b, c, d, I, sigma, T, n, V0, U0)
V = nan(n, T+1);
V(:, 1) = V0;
U = nan(n, T+1);
U(:, 1) = U0;
for t = 2:T+1
    spike = V(:, t-1) >= 30;
    V(:, t) = c * spike ... +
        + (V(:, t-1) + 0.1 * (0.04*V(:, t-1).^2+5*V(:, t-1)+140-U(:, t-1)+I)) .* ~spike ... +
        + sigma * randn(n, 1);
    U(:, t) = (U(:, t-1) + d) .* spike ... +
        + (U(:, t-1) + 0.1 * (a*b*V(:, t-1) - a*U(:, t-1))) .* ~spike;
end

function [R2_lin_rec, R2_nonlin_rec] = R2_gen(v_bar, u_bar, v_bar_diff, h_vals)
% Train-test separation
T = numel(v_bar);
T_test = round(0.5 * T);
test_ind = randperm(T, T_test);
train_ind = setdiff(1:T, test_ind);
v_bar_train = v_bar(train_ind);
u_bar_train = u_bar(train_ind);
v_bar_diff_train = v_bar_diff(train_ind);
v_bar_test = v_bar(test_ind);
u_bar_test = u_bar(test_ind);
v_bar_diff_test = v_bar_diff(test_ind);

% Linear regression
mdl = fitlm([v_bar_train; u_bar_train]', v_bar_diff_train');
save mdl.mat mdl
v_bar_diff_test_hat = predict(mdl, [v_bar_test; u_bar_test]')';
R2_denom = sum((v_bar_diff_test - mean(v_bar_diff_test)).^2);
R2_lin_rec = 1 - sum((v_bar_diff_test - v_bar_diff_test_hat).^2) / R2_denom;

% Nonlinear regression
n_h = numel(h_vals);
R2_nonlin_rec = nan(n_h, 1);
for i_h = 1:n_h
    h = h_vals(i_h);
    v_bar_diff_test_hat = LL_est([v_bar_train; u_bar_train]', v_bar_diff_train', ...
        [v_bar_test; u_bar_test]', [], h)';
    R2_nonlin_rec(i_h) = 1 - sum((v_bar_diff_test - v_bar_diff_test_hat).^2) / R2_denom;
end