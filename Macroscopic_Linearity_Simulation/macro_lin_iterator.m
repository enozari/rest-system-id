function macro_lin_iterator(section, run_on_cluster)
%MACRO_LIN_ITERATOR Code for iterating over macro_lin and obtaining and plotting
% aggregate statistics of its outputs.
% 
%   macro_lin_iterator(section, run_on_cluster) will run the specific
%   section of the code below either on a cluster, or locally.
%   run_on_cluster is a binary flag to determine if the code is to be run
%   on a cluster with Sun Grid Engine queuing system. section can take
%   values 1, 2, 3, 3.5, 4, 4.5, 5, 6, 7, or 8. See each section for
%   details. The general rule is that odd section numbers (including 3.5)
%   are for generating data and even section numbers (including 4.5) are
%   for plotting the results of the preceding section after it has been
%   completed.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

if nargin < 2 || isempty(run_on_cluster)
    run_on_cluster = 1;
end

% Adding the parent directory and its sub-directories to the path
full_filename = mfilename('fullpath');
addpath(genpath(full_filename(1:strfind(full_filename, 'Macroscopic Linearity Simulation')-2)))

if ~exist('macro_lin_data', 'dir')
    mkdir macro_lin_data
end

switch section
    case 1                                                                  % For sweeping over fpass with varying values of p
        sweep = 'fpass';                                                    % What to seep over
        p_vals = [0 0.25 0.5 1 2];                                          % the values of p to be used
        n_rep = 50;                                                         % Number of repetitions for each value of p
        if run_on_cluster
            write_macro_lin_sh(section);                                    % Function to write the shell code macro_lin.sh for submitting SGE jobs.
            system('chmod +x macro_lin.sh');                                % Making the shell code executable
            for p = p_vals
                system(['qsub -l s_vmem=64G -l h_vmem=64G ./macro_lin.sh ' sweep ' ' num2str(p) ' ' num2str(n_rep)]); % Submitting jobs to the cluster.
            end
        else
            for p = p_vals
                macro_lin_iterator_inner(section, p, n_rep)                 % Running the jobs locall and sequentially 
            end
        end
    case 2                                                                  % For harvesting and plotting the results of section 1. Section 1 must have completed.
        % Re-setting decision variables. Should be consistent with Section
        % 1 and/or macro_lin
        fpass_vals = logspace(-4, 0, 13);
        n_fpass = numel(fpass_vals);
        p_vals = [0 0.25 0.5 1 2];
        n_p = numel(p_vals);
        n_rep = 50;
        
        % Loading the results
        R2_lin_rec_rec = nan(n_rep, n_fpass, n_p);
        R2_nonlin_rec_rec = nan(n_rep, n_fpass, n_p);
        for i_p = 1:n_p
            p = p_vals(i_p);
            load(['macro_lin_data/section1_p' num2str(p) '.mat'], 'R2_lin_rec', 'R2_nonlin_rec');
            R2_lin_rec_rec(:, :, i_p) = R2_lin_rec;
            R2_nonlin_rec_rec(:, :, i_p) = R2_nonlin_rec;
        end
        R2_lin_rec = R2_lin_rec_rec;
        R2_nonlin_rec = R2_nonlin_rec_rec;
        
        % Graphics
        hf = figure;
        hf.Position(4) = hf.Position(4) - 50;
        boxplot(R2_lin_rec(:, end-4:end, p_vals == 0), 'PlotStyle', 'compact', 'Whisker', inf, ...
            'Colors', matlab_green, 'Positions', fpass_vals(end-4:end)/1.1)
        hold on
        boxplot(R2_nonlin_rec(:, end-4:end, p_vals == 0), 'PlotStyle', 'compact', 'Whisker', inf, ...
            'Colors', matlab_yellow, 'Positions', fpass_vals(end-4:end)*1.1)
        ha = gca;
        boxplot_magnify(ha.Children, 4)
        set(gca, 'xtick', fpass_vals(end-4:end), 'xticklabel', {'', '$10^{-1}$', '', '', '$10^0$'})
        ha.Position(3) = ha.Position(3) - 0.1;
        ha.Position(1) = ha.Position(1) + 0.1;
        ha.Position(4) = ha.Position(4) - 0.1;
        ha.Position(2) = ha.Position(2) + 0.1;
        hx = xlabel('$F_{\rm cutoff}$', 'Interpreter', 'latex');
        ylabel('$R^2$', 'Interpreter', 'latex')
        set(gca, 'xscale', 'log', 'xdir', 'reverse', 'fontsize', 35, 'TickLabelInterpreter', 'latex')%, ...
        xlim([fpass_vals(end-4)/1.2 fpass_vals(end)*1.2])
        ylim([0.87 1.01])
        hx.Position(2) = hx.Position(2) - 30;
        hb1 = bar(-1, 0, 'facecolor', matlab_green);
        hb2 = bar(-1, 0, 'facecolor', matlab_yellow);
        hl = legend([hb1 hb2], {'Linear', 'Nonlinear'});
        set(hl, 'fontsize', 35, 'Interpreter', 'latex')
        grid on
        hf.Color = 'w';
        export_fig macro_lin_iterator_section2_boxplot.eps
                
        hf = figure;
        R2_diff_rec = R2_nonlin_rec - R2_lin_rec;
        R2_diff_means = permute(mean(R2_diff_rec, 1), [2 3 1]);
        R2_diff_stds = permute(std(R2_diff_rec, [], 1), [2 3 1]);
        errorbar(repmat(fpass_vals', 1, n_p), R2_diff_means, R2_diff_stds/sqrt(n_rep), '.-', ...
            'linewidth', 4, 'CapSize', 0, 'MarkerSize', 35)
        xlabel('$f_{\rm cutoff}$', 'Interpreter', 'latex')
        ylabel('$R^2_{\rm NL} - R^2_{\rm L}$', 'Interpreter', 'latex')
        ha = gca;
        set(ha, 'xscale', 'log', 'xdir', 'reverse', 'fontsize', 35, 'ticklabelinterpreter', 'latex')
        legends = arrayfun(@(p)['$p = ' num2str(p) '$'], p_vals, 'UniformOutput', 0);
        hl = legend(ha, legends);
        set(hl, 'fontsize', 20, 'interpreter', 'latex')
        ylim([-0.005 0.1])
        colors = color_sweep(matlab_blue, 2, 2, [], 0.3);
        for i_p = 1:n_p
            ha.Children(i_p).Color = colors(i_p, :);
        end
        grid on
        hf.Color = 'w';
        export_fig macro_lin_iterator_section2_errorbar.eps

        hf = figure;
        min_freq = 5e-4;
        x_rec = cell(n_p, 1);
        PSD_rec = cell(n_p, 1);
        PSD_wspan_rec = cell(n_p, 1);
        hp = nan(n_p, 1);
        for i_p = 1:n_p
            p = p_vals(i_p);
            [~, ~, ~, x] = macro_lin('fpass', p, 0, 0);
            [PSD, PSD_wspan] = pwelch(x, numel(x)/20);
            PSD_wspan = PSD_wspan / pi;
            if PSD_wspan(2) > min_freq
                PSD_wspan = [PSD_wspan(1); min_freq; PSD_wspan(2:end)];
                X = fft(x);
                X_wspan = linspace(0, 2, numel(x)+1);
                X_wspan(end) = [];
                PSD = [PSD(1); interp1(X_wspan, abs(X).^2/numel(x)/pi, min_freq); PSD(2:end)];
            end
            PSD = PSD / max(PSD(PSD_wspan >= min_freq));
            x_rec{i_p} = x;
            PSD_rec{i_p} = PSD;
            PSD_wspan_rec{i_p} = PSD_wspan;
            hp(i_p) = loglog(PSD_wspan_rec{i_p}, PSD_rec{i_p}, 'color', colors(n_p+1-i_p, :), 'linewidth', 2);
            hold on
        end
        xlim([min_freq 1])
        ylim([1e-12 1e1])
        xlabel('$f$', 'Interpreter', 'latex')
        ylabel('PSD (a.u.)', 'Interpreter', 'latex')
        set(gca, 'XTick', 10.^(-3:0), 'YTick', 10.^(-10:5:0), 'TickLabelInterpreter', 'latex', 'FontSize', 35)
        wspan_mid = [1e-2 1e-1];
        PSD_mid = interp1(PSD_wspan, PSD, wspan_mid);
        loglog(wspan_mid, PSD_mid, 'LineWidth', 3, 'Color', 'k')
        annotation(hf, 'textbox', [0.301 0.3595 0.32 0.1642], 'FontSize', 35, ...
            'String', 'PSD $\propto \frac{1}{f^{2p}}$', 'LineStyle', 'none', 'Interpreter', 'latex')
        grid on
        hf.Color = 'w';
        export_fig macro_lin_iterator_section2_psd.eps
            
    case 3                                                                  % For sweeping over nave with varying values of p. Ensure uniform_cc = 0 in macro_lin.
        sweep = 'nave';
        p_vals = [0 0.25 0.5 1 2];
        n_rep = 100;                                                        % Number of repetitions for each value of p
        if run_on_cluster
            write_macro_lin_sh(section);                                    % Function to write the shell code macro_lin.sh for submitting SGE jobs.
            system('chmod +x macro_lin.sh');                                % Making the shell code executable
            for p = p_vals
                system(['qsub -l s_vmem=64G -l h_vmem=64G ./macro_lin.sh ' sweep ' ' num2str(p) ' ' num2str(n_rep)]); % Submitting jobs to the cluster.
            end
        else
            for p = p_vals
                macro_lin_iterator_inner(section, p, n_rep)                 % Running the jobs locally and sequentially
            end
        end
    case 3.5                                                                % Same as section 3 but for uniform_cc = 1
        sweep = 'nave';
        n_rep = 10;
        p = [];
        if run_on_cluster
            write_macro_lin_sh(section);
            system('chmod +x macro_lin.sh');
            system(['qsub -l s_vmem=64G -l h_vmem=64G ./macro_lin.sh ' sweep ' ' num2str(p) ' ' num2str(n_rep)]);
        else
            macro_lin_iterator_inner(section, p, n_rep)
        end
    case 4                                                                  % For harvesting and plotting the results of section 3. Section 3 must have completed.
        % Re-setting decision variables. Should be consistent with Section
        % 3 and/or macro_lin
        nave_vals = unique(round(logspace(0, 2, 9)));
        n_nave = numel(nave_vals);
        p_vals = [0 0.25 0.5 1 2];
        n_p = numel(p_vals);
        n_rep = 100;
        
        % Loading the results
        R2_lin_rec_rec = nan(n_rep, n_nave, n_p);
        R2_nonlin_rec_rec = nan(n_rep, n_nave, n_p);
        for i_p = 1:n_p
            p = p_vals(i_p);
            load(['macro_lin_data/section3_p' num2str(p) '.mat'], 'R2_lin_rec', 'R2_nonlin_rec');
            R2_lin_rec_rec(:, :, i_p) = R2_lin_rec;
            R2_nonlin_rec_rec(:, :, i_p) = R2_nonlin_rec;
        end
        R2_lin_rec = R2_lin_rec_rec;
        R2_nonlin_rec = R2_nonlin_rec_rec;
                
        % Graphics
        hf = figure;
        plot_p_ind = 1:5;
        R2_diff_rec = R2_nonlin_rec - R2_lin_rec;
        R2_diff_means = permute(mean(R2_diff_rec, 1), [2 3 1]);
        R2_diff_stds = permute(std(R2_diff_rec, [], 1), [2 3 1]);
        errorbar(repmat(nave_vals', 1, numel(plot_p_ind)), R2_diff_means(:, plot_p_ind), ...
            R2_diff_stds(:, plot_p_ind)/sqrt(n_rep), '.-', 'linewidth', 4, 'CapSize', 0, 'MarkerSize', 35)
        xlabel('$N_{\rm ave}$', 'Interpreter', 'latex')
        ylabel('$R^2_{\rm NL} - R^2_{\rm L}$', 'Interpreter', 'latex')
        ha = gca;
        set(ha, 'xscale', 'log', 'fontsize', 35, 'ticklabelinterpreter', 'latex')
        legends = arrayfun(@(p)['$p = ' num2str(p) '$'], p_vals(plot_p_ind), 'UniformOutput', 0);
        legend(ha, legends, 'fontsize', 30, 'Interpreter', 'latex')
        colors = color_sweep(matlab_blue, 2, 2, [], 0.3);
        for i_p = 1:numel(plot_p_ind)
            ha.Children(i_p).Color = colors(i_p, :);
        end
        grid on
        ylim([-0.002 0.12])
        hf.Color = 'w';
        export_fig macro_lin_iterator_section4_errorbar.eps
        
        hf = figure;
        hold on
        rhomax = 0.5;
        dmin = 1e-3;
        rho = @(d, p)rhomax .* (d <= dmin) + rhomax*dmin^p./d.^p .* (d > dmin);
        d_vals = linspace(0, 1, 1e3);
        for i_p = 1:n_p
            p = p_vals(i_p);
            plot(d_vals, rho(d_vals, p), 'linewidth', 5, 'color', colors(n_p+1-i_p, :))
        end
        xlabel('dist (a.u.)', 'Interpreter', 'latex')
        ylabel('Pearson $\rho$', 'Interpreter', 'latex')
        grid on
        hf.Color = 'w';
        set(gca, 'fontsize', 35, 'ticklabelinterpreter', 'latex')
        ylim([0 0.6])
        annotation(hf, 'textbox', [0.4331 0.4809 0.2275 0.1738], 'FontSize', 35, ...
            'String', '$\rho \propto \frac{1}{{\rm dist}^p}$', 'LineStyle', 'none', 'Interpreter', 'latex')
        hf.Color = 'w';
        export_fig macro_lin_iterator_section4_rho.eps
    case 4.5                                                                % For harvesting and plotting the results of section 3.5. Section 3.5 must have completed.
        % Re-setting decision variables. Should be consistent with
        % macro_lin
        nave_vals = 1:6;
        
        % Loading the results
        load macro_lin_data/section3_p.mat R2_lin_rec R2_nonlin_rec
        
        % Graphics
        hf = figure;
        hf.Position(4) = hf.Position(4) - 50;
        boxplot(R2_lin_rec, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_green, ...
            'Positions', nave_vals+0.15)
        hold on
        boxplot(R2_nonlin_rec, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_yellow, ...
            'Positions', nave_vals-0.15)
        ha = gca;
        boxplot_magnify(ha.Children, 4)
        set(gca, 'xtick', nave_vals, 'xticklabel', arrayfun(@num2str, nave_vals, 'UniformOutput', 0))
        hx = xlabel('$N_{\rm ave}$', 'Interpreter', 'latex');
        ylabel('$R^2$', 'Interpreter', 'latex')
        ha.FontSize = 35; 
        ha.TickLabelInterpreter = 'latex';
        ha.Position(4) = ha.Position(4) - 0.1;
        ha.Position(2) = ha.Position(2) + 0.1;
        xlim([nave_vals(1)-0.5 nave_vals(end)+0.5])
        ylim([0.85 1.01])
        hx.Position(2) = hx.Position(2) - 30;
        hb1 = bar(-1, 0, 'FaceColor', matlab_green);
        hb2 = bar(-1, 0, 'FaceColor', matlab_yellow);
        legend([hb1 hb2], {'Linear', 'Nonlinear'}, 'fontsize', 35, 'interpreter', 'latex')
        grid on
        hf.Color = 'w';
        export_fig macro_lin_iterator_section4_5_boxplot.eps
        
    case 5                                                                  % For sweeping over SNR.
        sweep = 'SNR';
        p = 0;                                                              % The value of p is irrelevant for sweep = 'SNR'
        n_rep = 100;                                                        % Number of repetitions for each value of p
        if run_on_cluster
            write_macro_lin_sh(section);                                    % Function to write the shell code macro_lin.sh for submitting SGE jobs.
            system('chmod +x macro_lin.sh');                                % Making the shell code executable
            system(['qsub -l s_vmem=64G -l h_vmem=64G ./macro_lin.sh ' sweep ' ' num2str(p) ' ' num2str(n_rep)]); % Submitting the job to the cluster
        else
            macro_lin_iterator_inner(section, p, n_pre)                     % Running the job locally
        end
    case 6                                                                  % For harvesting and plotting the results of section 5. Section 5 must have completed.
        % Re-setting decision variables. Should be consistent with
        % macro_lin
        SNR_vals = logspace(0, 2, 9);
        
        % Loading the results
        load macro_lin_data/section5_p0.mat R2_lin_rec R2_nonlin_rec
        
        % Graphics
        hf = figure;
        hf.Position(4) = hf.Position(4) - 50;
        boxplot(R2_lin_rec, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_green, ...
            'Positions', SNR_vals/1.1)
        hold on
        boxplot(R2_nonlin_rec, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_yellow, ...
            'Positions', SNR_vals*1.1)
        ha = gca;
        boxplot_magnify(ha.Children, 4)
        set(gca, 'xtick', SNR_vals, 'xticklabel', {'$10^0$', '', '', '', '$10^1$', '', '', '', '$10^2$'})
        hx = xlabel('SNR', 'Interpreter', 'latex');
        ylabel('$R^2$', 'Interpreter', 'latex')
        set(gca, 'xscale', 'log', 'xdir', 'reverse', 'fontsize', 35, 'TickLabelInterpreter', 'latex')
        ha.Position(4) = ha.Position(4) - 0.1;
        ha.Position(2) = ha.Position(2) + 0.1;
        xlim([SNR_vals(1)/1.2 SNR_vals(end)*1.2])
        ylim([0 1.05])
        hx.Position(2) = hx.Position(2) - 30;
        hb1 = bar(-1, 0, 'facecolor', matlab_green);
        hb2 = bar(-1, 0, 'facecolor', matlab_yellow);
        legend([hb1 hb2], {'Linear', 'Nonlinear'}, 'Location', 'southwest', 'fontsize', 35, ...
            'Interpreter', 'latex')
        grid on
        hf.Color = 'w';
        export_fig macro_lin_iterator_section6.eps
        
    case 7                                                                  % For sweeping over dim
        sweep = 'dim';
        p = 0;                                                              % The value of p is irrelevant for sweep = 'SNR'
        n_rep = 100;                                                        % Number of repetitions for each value of p
        if run_on_cluster
            write_macro_lin_sh(section);                                    % Function to write the shell code macro_lin.sh for submitting SGE jobs.
            system('chmod +x macro_lin.sh');                                % Making the shell code executable
            system(['qsub -l s_vmem=64G -l h_vmem=64G ./macro_lin.sh ' sweep ' ' num2str(p) ' ' num2str(n_rep)]); % Submitting the job to the cluster
        else
            macro_lin_iterator_inner(section, p, n_pre)                     % Running the job locally
        end
    case 8                                                                  % For harvesting and plotting the results of section 7. Section 7 must have completed.
        % Re-setting decision variables. Should be consistent with
        % macro_lin
        dim_vals = unique(round(logspace(0, 2, 11)));
        
        % Loading the results
        load macro_lin_data/section7_p0.mat R2_lin_rec R2_nonlin_rec h_opt_rec
        
        % Graphics
        hf = figure;
        hf.Position(4) = hf.Position(4) - 50;
        boxplot(R2_lin_rec, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_green, ...
            'Positions', dim_vals*1.05)
        hold on
        boxplot(R2_nonlin_rec, 'PlotStyle', 'compact', 'Whisker', inf, 'Colors', matlab_yellow, ...
            'Positions', dim_vals/1.05)
        ha = gca;
        boxplot_magnify(ha.Children, 3)
        set(gca, 'xtick', 10.^(0:2), 'xticklabel', {'$10^0$', '$10^1$', '$10^2$'}, ...
            'TickLabelInterpreter', 'latex')
        hx = xlabel('Dimension', 'Interpreter', 'latex');
        ylabel('$R^2$', 'Interpreter', 'latex')
        set(gca, 'xscale', 'log', 'fontsize', 35)
        ha.Position(4) = ha.Position(4) - 0.1;
        ha.Position(2) = ha.Position(2) + 0.1;
        xlim([dim_vals(1)/1.2 dim_vals(end)*1.2])
        hx.Position(2) = hx.Position(2) - 30;
        hb1 = bar(-1, 0, 'facecolor', matlab_green);
        hb2 = bar(-1, 0, 'facecolor', matlab_yellow);
        legend([hb1 hb2], {'Linear', 'Nonlinear'}, 'FontSize', 30, 'Interpreter', 'latex')
        grid on
        hf.Color = 'w';
        export_fig macro_lin_iterator_section8_boxplot.eps
        
        hf = figure;
        hf.Position(4) = hf.Position(4) - 50;
        n_rep = size(h_opt_rec, 1);
        h_opt_means = mean(h_opt_rec, 1);
        h_opt_stds = std(h_opt_rec, [], 1);
        errorbar(dim_vals, h_opt_means, h_opt_stds/sqrt(n_rep), '.-', 'linewidth', 4, 'CapSize', 0, ...
            'MarkerSize', 35)
        xlabel('Dimension', 'Interpreter', 'latex')
        ylabel('Opt Win Size ($h$)', 'Interpreter', 'latex')
        ha = gca;
        set(ha, 'xscale', 'log', 'yscale', 'log', 'fontsize', 35, 'ticklabelinterpreter', 'latex')
        grid on
        hf.Color = 'w';
        export_fig macro_lin_iterator_section8_errorbar.eps
end
end

%% Auxiliary functions
function write_macro_lin_sh(section)                                        % This function writes the shell script macro_lin.sh to the same directory. This is used for running jobs on a cluster. The macro_lin.sh could be directly written, but in this way all the code is accessible from MATLAB. Note that this is unnecessary if the jobs are going to be run locally.
s = ["#!/bin/bash";
    "matlab -r ""\";
    "   sweep = '$1'; \";
    "   p = $2; \";
    "   n_rep = $3; \";
    "   R2_lin_rec = cell(n_rep, 1); \";
    "   R2_nonlin_rec = cell(n_rep, 1); \";
    "   h_opt_rec = cell(n_rep, 1); \";
    "   for i_rep = 1:n_rep, \";
    "       [R2_lin_rec{i_rep}, R2_nonlin_rec{i_rep}, h_opt_rec{i_rep}] = macro_lin(sweep, p, 1, 0); \";
    "   end, \";
    "   R2_lin_rec = cell2mat(R2_lin_rec); \";
    "   R2_nonlin_rec = cell2mat(R2_nonlin_rec); \";
    "   h_opt_rec = cell2mat(h_opt_rec); \";
    "   save(['macro_lin_data/section" + num2str(section) + "_p' num2str(p) '.mat'], 'R2_lin_rec', 'R2_nonlin_rec', 'h_opt_rec'); \";
    "   exit"""];
fileID = fopen('macro_lin.sh', 'w');
fprintf(fileID, '%s\n', s);
end