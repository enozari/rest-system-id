function subj_org_ieeg(ss_factor, n_segment_per_recording)
%SUBJ_ORG_IEEG segmenting and subsampling iEEG data for system
% identification studies reported in 
% E. Nozari et. al., "Is the brain macroscopically linear? A system
% identification of resting state dynamics", 2021.
% 
%   The code assumes to be available in a base_address organized first
%   by subjects and then (within the folder for each subject) recordings
%   per subject. In the above paper, data files were each 5 minutes long
%   sampled originally at 500Hz.
% 
%   Input arguments:
% 
%   ss_factor: subsampling factor (integer greater or equal to 1).
% 
%   n_segment_per_recording: number of segments per (5-minute) recording.
%   In the above paper, the following values were used:
%   n_segment_per_recording = 30 if ss_factor == 1
%   n_segment_per_recording = 5 if ss_factor == 5
%   n_segment_per_recording = 1 if ss_factor == 25

base_address = '/Volumes/WD My Passport 25E2 Media/Erfan/Data/RAM Resting State/';
listing = struct2cell(dir(base_address));
subjs = listing(1, :)';
subjs = subjs(cellfun(@(name)name(1) ~= '.', subjs));
recordings = cell(size(subjs));
for i = 1:numel(recordings)
    listing = struct2cell(dir([base_address subjs{i}]));
    recordings{i} = cellfun(@(c)[subjs{i} '/' c], listing(1, 3:end), 'UniformOutput', 0);
end
recordings = [recordings{:}]';
n_recording = numel(recordings);

n_segment = n_recording * n_segment_per_recording;
rng(1);
ind_segment = randperm(n_segment);
if ~exist(['rs_5min/rand_segments_' num2str(ss_factor)], 'dir')
    mkdir(['rs_5min/rand_segments_' num2str(ss_factor)])
end
for i_segment = 1:n_segment
    disp(i_segment)
    [i_recording, i_segment_per_recording] = ind2sub([n_recording, n_segment_per_recording], ind_segment(i_segment));
    load([base_address recordings{i_recording}], 'data')
    Y = data.trial{1}(:, end/n_segment_per_recording*(i_segment_per_recording-1)+1:end/n_segment_per_recording*i_segment_per_recording);
    Y = Y(:, 1:ss_factor:end);
    save(['rs_5min/rand_segments_' num2str(ss_factor) '/Y_' num2str(i_segment) '.mat'], 'Y')
end
end