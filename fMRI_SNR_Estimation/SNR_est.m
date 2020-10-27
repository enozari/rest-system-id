function SNR_est(section, varargin)
%SNR_EST This function estimates scannar signal to noise ratio (SNR) for
% fMRI data as shown in the supplementary information of the paper:
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2020.
%
%   Input Arguments
% 
%   section: which section of the code to run. The sections should be run
%   in sequence, preferrably on a cluster, each after the previous has
%   finished running. Section 1 selects random scans and submits the
%   proprocessing jobs for each scan to the cluster (or runs them serially
%   on the same machine if run_on_cluster == 0). Section 1.5 is only to be
%   run internally by Section 1. When finished, call Section 2 to collect
%   the results of Section 1 and estimate the SNRs. When finished, call
%   Section 3 to plot the results as reported in the above paper.
% 
%   varargin: only used internally by Section 1.5.
% 
%   Copyright (C) 2020, Erfan Nozari
%   All rights reserved.

run_on_cluster = 1;                                                         % Switch to determine if this code is run on a SGE-managed cluster, where different scans can be processed in parallel, or not, where all scans will be run in sequence.

load subjs.mat subjs                                                        % The list of all 700 HCP subjects used in the main study
n_subj = numel(subjs);
scans = {'1_LR', '1_RL', '2_LR', '2_RL'};                                   % The list of the four scans recorded from each subject
n_scan = numel(scans);

n_rand = 50;                                                                % Number of random scans to be selected for SNR estimation (out of the total 2800 scans)
rng(1)
rand_subjs = subjs(randperm(n_subj, n_rand));                               % Selecting n_rand subjects uniformly at random
rand_scans = scans(randi(n_scan, n_rand, 1));                               % Selecting 1 scan per subject uniformly at random

switch section
    case 1
        system('rm -f -r SNR_est');                                         % Removing any leftovers from previous runs of this function
        mkdir SNR_est                                                       % Directory to store the intermediate files
        if run_on_cluster
            write_SNR_est_sh()                                              % Auxiliary function written below that generates the shell code used for submitting the jobs to the SGE cluster
            system('chmod +x SNR_est.sh');                                  % Making the just created shell code executable
            for i_rand = 1:n_rand
                subj = rand_subjs{i_rand};
                scan = rand_scans{i_rand};
                system(['qsub -l s_vmem=64G -l h_vmem=64G ./SNR_est.sh ' subj ' ' scan]); % Submitting each job to the cluster, which will run Section 1.5 on the corresponding scan
            end
        else
            for i_rand = 1:n_rand
                subj = rand_subjs{i_rand};
                scan = rand_scans{i_rand};
                SNR_est(1.5, subj, scan);                                   % Corresponding line of code if all the scans are to be processed locally in serial
            end
        end
        
    case 1.5                                                                % This section, only to be run internally by Section 1, performs a series of preprocessing steps on each scan and same subject's T1 scan using FSL shell commands
        subj = varargin{1};
        scan = varargin{2};
        
        cd SNR_est
        base_address = ['/cbica/projects/HCP_Data_Releases/HCP_1200/' subj '/unprocessed/3T']; % The base address for HCP data, change it to your respective base address
        system(['mcflirt -in ' base_address '/rfMRI_REST' scan '/' subj '_3T_rfMRI_REST' scan '.nii.gz ' ...
            '-out ' subj '_3T_rfMRI_REST' scan '_mcf']);                    % Intramodal motion correction of the fMRI time series
        system(['fslroi ' subj '_3T_rfMRI_REST' scan '_mcf.img ' ...
            subj '_3T_rfMRI_REST' scan '_mcf_frame1 0 1']);                 % Taking the first volume (first sample of each voxel) for visual inspection and selection of 10 voxels outside of the head
        system(['bet ' base_address '/T1w_MPR1/' subj '_3T_T1w_MPR1.nii.gz ' ...
            'temp_' subj '_3T_T1w_MPR1_brain -m']);                         % Brain extraction, used subsequently for gray matter extraction and registration to/from the fMRI image
        system(['fast -g -o temp_' subj '_3T_T1w_MPR1_brain_seg ' ...
            '-v temp_' subj '_3T_T1w_MPR1_brain.img']);                     % Extraction of the gray matter voxels
        system(['flirt -in ' subj '_3T_rfMRI_REST' scan '_mcf.img ' ...
            '-ref temp_' subj '_3T_T1w_MPR1_brain.img ' ...
            '-omat temp_' subj '_3T_rfMRI_mcf_to_T1w_brain.mat']);          % Linear registration of the motion corrected fMRI scan to the brain extracted T1 scan
        system(['convert_xfm -omat temp_' subj '_3T_T1w_brain_to_rfMRI_mcf.mat ' ...
            '-inverse temp_' subj '_3T_rfMRI_mcf_to_T1w_brain.mat']);       % Inverting the registration matrix
        system(['flirt -in temp_' subj '_3T_T1w_MPR1_brain_seg_seg_1.img ' ...
            '-out ' subj '_3T_T1w_MPR1_brain_GM_mask_invreg ' ...
            '-ref ' subj '_3T_rfMRI_REST' scan '_mcf.nii ' ...
            '-applyxfm -init temp_' subj '_3T_T1w_brain_to_rfMRI_mcf.mat']); % Inverse registration of the gray matter mask to the fMRI space
        system(['fslmaths ' subj '_3T_T1w_MPR1_brain_GM_mask_invreg.img ' ...
            '-sub 0.5 -bin ' subj '_3T_T1w_MPR1_brain_GM_mask_invreg']);    % Binarizing the inverse-registered gray matter mask
        system(['rm -f temp_' subj '*']);                                   % Removing all the intermediate temporary files
        cd ..
        
    case 2                                                                  % This section loads the results of Section 1 and, together with a pre-populated list of manually picked voxels based on the visualization of the ['subj '_3T_rfMRI_REST' scan '_mcf_frame1'] image, estimates the SNR for each subject
        outside_skull_ind = {                                               % The list of 10 manually picked voxels outside the head per subject-scan. Each triplets shows an x-y-z coordinate from FSLeyes.
            '100206', 1+[44 81 64; 45 102 37; 43 95 67; 63 66 69; 24 86 60; 85 22 60; 66 2 50; 16 2 22; 16 73 63; 71 76 63];
            '103111', 1+[48 95 56; 48 91 57; 34 69 69; 73 80 55; 22 58 71; 13 74 65; 70 74 61; 70 74 61; 43 6 69; 26 71 69];
            '118831', 1+[44 84 64; 44 10 69; 74 10 57; 14 10 45; 8 15 45; 77 84 45; 77 16 64; 20 80 64; 74 80 56; 78 91 35];
            '135528', 1+[44 85 59; 44 12 63; 68 12 56; 14 12 10; 81 92 10; 25 85 49; 69 85 51; 69 13 61; 18 13 11; 18 74 62];
            '136227', 1+[44 90 66; 44 3 71; 23 3 61; 25 79 63; 70 81 63; 84 81 52; 84 17 61; 11 17 66; 26 73 66; 74 68 66];
            '144428', 1+[44 86 67; 44 7 67; 70 76 57; 12 73 57; 86 73 26; 10 76 56; 80 62 56; 15 62 68; 77 27 68; 77 9 52];
            '150928', 1+[44 84 64; 18 84 46; 71 84 51; 9 76 51; 67 76 63; 67 89 48; 67 11 59; 17 76 59; 17 84 49; 80 84 37];
            '151425', 1+[44 90 64; 44 6 70; 68 6 58; 19 6 57; 19 74 64; 61 81 64; 48 89 63; 44 83 63; 44 7 66; 18 7 62];
            '151627', 1+[44 79 71; 44 6 66; 12 86 37; 68 86 52; 68 41 71; 53 66 71; 53 95 51; 16 81 51; 24 81 59; 24 61 71];
            '151930', 1+[44 95 59; 22 73 68; 8 73 56; 13 68 67; 74 66 67; 13 78 60; 78 18 60; 80 50 63; 65 78 63; 14 78 56];
            '153227', 1+[44 81 66; 44 11 67; 66 11 56; 75 11 12; 77 84 12; 76 70 54; 11 22 53; 44 91 53; 78 69 53; 2 62 14];
            '154835', 1+[44 87 62; 25 72 62; 25 85 53; 25 12 63; 69 70 63; 9 70 51; 72 70 64; 72 82 55; 16 70 55; 82 27 55];
            '154936', 1+[44 85 65; 26 85 55; 13 72 55; 20 11 59; 73 78 59; 73 12 63; 67 73 63; 67 86 54; 26 86 51; 83 67 51];
            '155938', 1+[44 84 69; 62 84 60; 18 80 60; 18 65 70; 7 65 61; 14 15 61; 75 11 61; 16 57 68; 83 57 60; 4 32 60];
            '156334', 1+[44 77 67; 74 77 51; 11 77 43; 74 87 43; 29 74 62; 29 12 64; 72 12 52; 20 83 52; 20 62 65; 68 69 65];
            '173435', 1+[44 85 66; 60 85 56; 20 85 53; 12 17 53; 83 19 53; 8 19 10; 63 88 55; 63 2 49; 3 45 49; 77 10 49];
            '175237', 1+[12 51 68; 13 40 67; 13 72 57; 79 22 57; 71 80 54; 12 72 54; 13 16 54; 80 16 39; 14 81 44; 14 68 61];
            '176542', 1+[11 85 35; 71 85 49; 71 67 60; 16 27 60; 86 27 17; 53 11 64; 68 11 53; 79 26 53; 5 26 9; 51 88 58];
            '177645', 1+[8 70 35; 72 70 62; 72 12 59; 4 40 59; 82 40 64; 14 65 64; 40 4 63; 27 12 63; 78 12 42; 78 66 54];
            '178950', 1+[44 5 67; 70 5 59; 70 95 25; 70 3 51; 9 15 51; 18 15 65; 80 81 52; 5 81 13; 74 14 63; 74 91 18];
            '180836', 1+[7 51 56; 78 15 56; 10 15 31; 16 93 31; 16 9 52; 75 78 52; 15 78 51; 12 15 51; 86 50 40; 8 75 40];
            '196346', 1+[4 6 35; 71 6 61; 71 97 21; 11 91 21; 39 97 54; 72 7 54; 4 17 54; 81 17 61; 23 81 62; 23 4 63];
            '198047', 1+[81 51 54; 64 7 54; 64 94 47; 64 97 9; 35 90 56; 35 8 68; 84 38 58; 2 38 30; 11 9 30; 75 13 50];
            '199352', 1+[9 51 63; 78 51 67; 70 80 53; 70 9 45; 7 22 45; 17 22 57; 10 68 57; 85 68 32; 85 18 62; 17 30 62];
            '202719', 1+[6 44 35; 77 44 67; 4 44 47; 14 6 47; 14 69 59; 6 54 59; 28 5 62; 2 5 12; 82 77 12; 68 77 54];
            '206828', 1+[5 51 61; 76 65 61; 76 6 51; 17 6 50; 10 79 50; 22 79 61; 43 85 61; 43 8 70; 12 41 68; 18 73 62];
            '231928', 1+[44 85 63; 19 67 63; 6 67 29; 13 26 63; 77 54 63; 23 74 63; 7 74 27; 35 70 71; 87 39 26; 12 39 66];
            '245333', 1+[44 68 70; 70 68 59; 6 68 25; 81 74 25; 12 8 25; 61 8 58; 61 95 50; 9 75 50; 80 75 9; 12 75 17];
            '318637', 1+[44 90 62; 20 90 50; 78 82 50; 8 74 50; 28 10 64; 28 86 55; 47 2 55; 50 82 64; 39 100 45; 6 15 45];
            '379657', 1+[44 6 62; 79 24 62; 9 24 54; 85 54 45 ; 61 98 32; 8 79 32; 76 79 49; 76 71 62; 20 14 62; 6 51 57];
            '389357', 1+[44 82 67; 20 62 68; 20 18 64; 10 18 40; 85 69 40; 6 69 27; 13 69 59; 78 61 59; 6 61 21; 77 7 21];
            '395251', 1+[44 9 62; 20 9 42; 20 90 17; 9 83 17; 64 83 56; 64 5 7; 48 81 65; 10 49 65; 3 49 46; 67 6 46];
            '517239', 1+[85 51 66; 18 64 66; 18 89 48; 18 57 70; 6 8 38; 87 21 38; 11 86 38; 11 11 61; 64 11 68; 64 94 51];
            '540436', 1+[9 10 35; 72 97 35; 72 13 62; 11 13 52; 84 19 52; 66 79 59; 66 2 10; 66 73 65; 14 26 65; 86 26 8];
            '553344', 1+[44 81 69; 44 7 64; 21 18 64; 73 18 68; 14 18 10; 14 71 64; 17 64 64; 17 12 15; 76 91 15; 76 70 62];
            '557857', 1+[5 51 17; 72 93 17; 72 9 58; 13 9 27; 6 88 27; 9 37 68; 78 77 46; 4 41 46; 81 15 46; 71 15 65];
            '570243', 1+[44 76 69; 20 21 69; 20 85 47; 46 85 63; 72 69 63; 72 7 52; 5 24 52; 9 24 5; 9 71 61; 76 57 61];
            '615744', 1+[4 51 26; 4 58 71; 17 47 71; 6 47 64; 51 95 10; 9 21 10; 78 97 10; 78 7 13; 13 83 43; 80 83 27];
            '654350', 1+[44 5 68; 44 97 57; 11 76 57; 86 76 24; 13 76 56; 13 10 18; 86 15 18; 13 15 63; 13 76 66; 76 16 66];
            '665254', 1+[8 69 35; 73 69 60; 73 80 56; 11 73 56; 11 7 45; 75 7 8; 15 7 57; 31 87 57; 31 6 63; 74 19 63];
            '680452', 1+[81 51 64; 81 13 65; 67 80 65; 67 99 15; 4 20 15; 83 8 15; 83 85 70; 7 85 47; 81 11 47; 64 11 67];
            '732243', 1+[44 91 63; 16 70 63; 83 70 34; 83 14 62; 10 14 8; 7 90 8; 20 90 60; 20 28 68; 70 19 68; 70 6 5];
            '818859', 1+[44 92 65; 6 52 65; 76 68 65; 5 68 61; 5 12 29; 82 95 29; 82 8 3; 12 8 59; 74 83 59; 4 83 16];
            '857263', 1+[6 81 35; 65 81 60; 65 5 65; 11 51 65; 11 78 57; 86 74 4; 86 70 62; 8 22 62; 79 22 69; 12 65 65];
            '861456', 1+[44 10 64; 22 10 45; 80 19 45; 20 19 62; 20 88 49; 11 91 49; 75 19 8; 75 81 61; 15 21 61; 66 21 66];
            '870861', 1+[14 51 65; 14 75 53; 76 66 53; 76 9 35; 16 9 53; 68 82 53; 68 96 19; 7 86 19; 7 69 65; 19 30 65];
            '877269', 1+[44 85 65; 23 71 65; 86 71 20; 13 23 60; 62 79 60; 62 93 43; 5 29 43; 77 29 68; 16 51 67; 67 5 9];
            '885975', 1+[8 13 35; 8 14 8; 81 32 8; 7 32 60; 47 103 8; 78 85 8; 18 85 56; 75 77 56; 11 77 44; 77 12 44];
            '894673', 1+[75 51 69; 31 80 63; 21 70 63; 21 6 50; 5 31 50; 85 31 12; 76 12 12; 62 12 64; 12 37 64; 83 37 59];
            '899885', 1+[44 81 66; 16 20 66; 5 20 47; 5 18 9; 77 18 17; 16 90 17; 45 91 58; 12 15 58; 12 69 50; 75 10 50]};
        
        cd SNR_est
        SNR = nan(n_rand, 1);
        for i_rand = 1:n_rand
            subj = rand_subjs{i_rand};
            scan = rand_scans{i_rand};
            
            rsfmri_filename = [subj '_3T_rfMRI_REST' scan '_mcf'];
            V_rsfmri = double(niftiread(rsfmri_filename));
            V_rsfmri_var = var(V_rsfmri, [], 4);                            % Temporal variance of each fMRI voxel

            xyz = outside_skull_ind{ismember(outside_skull_ind(:, 1), subj), 2};
            N2 = mean(V_rsfmri_var(sub2ind(size(V_rsfmri_var), xyz(:, 1), xyz(:, 2), xyz(:, 3)))); % Noise variance estimate, calculated from the voxels outside the head

            gm_filename = [subj '_3T_T1w_MPR1_brain_GM_mask_invreg'];
            V_gm = logical(niftiread(gm_filename));
            S2pN2 = mean(V_rsfmri_var(V_gm));                               % Signal plust noise variance estimate, calculated from gray matter voxels

            SNR(i_rand) = sqrt((S2pN2 - N2) / N2);                          % Signal to noise ratio
        end
        cd ..
        
        save SNR_est_data.mat SNR n_rand rand_subjs rand_scans outside_skull_ind
        
    case 3                                                                  % Plotting the graphics
        load SNR_est_data.mat SNR
        figure('color', 'w')
        histogram(SNR)
        xlabel('SNR', 'Interpreter', 'latex', 'FontSize', 20)
        ylabel('Count', 'Interpreter', 'latex', 'FontSize', 20)
        set(gca, 'ticklabelinterpreter', 'latex', 'fontsize', 20)
        ylim([0 20])
        hold on
        ylims = get(gca, 'ylim');
        plot(mean(SNR)*[1 1], ylims, 'k--', 'linewidth', 5)
        annotation(gcf, 'textbox', [0.4438 0.7547 0.1775 0.1095], 'String', '$\rm SNR_{\rm ave} = 6.5$', ...
            'LineStyle', 'none', 'Interpreter', 'latex', 'FontSize', 20);
        export_fig SNR.eps
end
end

%% Auxiliary functions
function write_SNR_est_sh                                                   % This function writes the shell script SNR_est.sh to the same directory. This is used for running jobs on a cluster. The SNR_est.sh could be directly written in a text file, but in this way all the code is accessible from MATLAB. Note that this is unnecessary if the jobs are going to be run locally.
s = ["#!/bin/bash";
    "matlab -r ""SNR_est(1.5, '$1', '$2'); exit"""];
fileID = fopen('SNR_est.sh', 'w');
fprintf(fileID, '%s\n', s);
fclose(fileID);
end