%% Example for metric learning with COMET 
% ("Metric Learning One Feature at a Time", 
%   Y. Atzmon, U. Shalit, G. Chechik, 2015 )
%
% This example learns a metric for the protein (LIBSVM) dataset and
% RCV1_4 with 5K features, using sparse COMET.
% To run this example with the protein dataset, you should download and
% install LIBSVM matlab package and add it to your matlab path
%
% The COMET method also requires Suitesparse (4.4.5+) Cholesky
% solver. See README.md for installation instructions 

%% init
clear
initpaths % Edit initpaths.m according to your respective environment.

% Tcnt = Number of triplets
% seed = Randomization seed.
% num_d_repeats = Number of times repeating the iterations on the d coords.
% max_step_size = Maximal step size to take (step size can be smaller if  
%                                           bounded by the Schur complement).
% logdet_weight = Logdet regularizer coefficient.
% sparse_weight = Sparse regularizer coefficient.
% frob_weight = Frobenius regularizer coefficient.
% step_size_bound_weight = A multiplicative weight to stay off nulling
%                          of eigenvalues (stay off touching the PSD
%                          cone boundary), emprically chosen to be 0.3 .

%% Set parameters:

dataset_name = 'protein';
% dataset_name = 'RCV1_4_5K';

% Default values:
cfg.step_size_bound_weight = 0.3; % Default value is 0.3,
                                  % emprically found to yield good
                                  % results while maintaining
                                  % numerical stability

cfg.show_progress_prints_flag = true; % set to false supresses verbosity

% Choosing the dataset and its hyper params

if strcmp(dataset_name, 'protein')
    %%% Load and set the hyper params for the Protein dataset
    cfg.Tcnt = 20e3;
    cfg.seed = 0;
    cfg.num_d_repeats = 15;
    cfg.max_step_size = 1e-3;
    cfg.logdet_weight = 5;
    cfg.frob_weight = 0;
    
    %% Load dataset
    [labels, samples ] = libsvmread('data/protein_libsvm_dataset');
    labels = labels + 1; % Labels must be in the range 1, 2, ... for the triplets generation


elseif strcmp(dataset_name, 'RCV1_4_5K')
    %%% Load and set the hyper params for the RCV1_4 with 5K features dataset
    cfg.Tcnt = 100e3;
    cfg.seed = 0;
    % Off diagonal training hyper params
    cfg.num_d_repeats = 8;
    cfg.max_step_size = .01;
    cfg.logdet_weight = 0.5;
    cfg.frob_weight = 0;

    %% Load dataset
    data_s = load('data/rcv1_4_infogain5000.mat', 'labels', 'data');
    labels = data_s.labels;
    samples = data_s.data;

else
    error('Unknown dataset_name: %s\n', dataset_name);
end

% Split dataset:
split_ratio = 0.8;
rng('default')
rng(cfg.seed)
[samples_trn, samples_tst, labels_trn, labels_tst] = ...
    split_by_ratio(samples, labels, split_ratio);


%% Generate triplets:
filtered_printf(cfg.show_progress_prints_flag, 'Generating a triplets set\n');
[ triplets_trn, triplets_trn_Q, triplets_trn_diff_mat ] = ...
    comet_generate_triplets( cfg.Tcnt, samples_trn, labels_trn, cfg.seed);
filtered_printf(cfg.show_progress_prints_flag, 'Done generating a triplets set\n');

%% Train the metric learning algorithm:
tic
[metric, inv_metric] =  train_dense_comet(cfg, triplets_trn, triplets_trn_Q, ...
    triplets_trn_diff_mat);
toc

%% Evaluate precision @k performance:

fprintf('\n\nRESULTS:\n========\n');
% COMET performance:
[mAP_metric_tst_set, prec_at_all_k_metric_tst_set, AUC_metric_tst_set] =...
    evaluate_precision(samples_tst, labels_tst, metric);

fprintf('Dense COMET    : mean AP     tst_set = %f\n', mAP_metric_tst_set);
fprintf('Dense COMET    : AUC         tst_set = %f\n', AUC_metric_tst_set);
fprintf('Dense COMET    : precision@5 tst_set = %f\n', prec_at_all_k_metric_tst_set(5));
fprintf('\n');

% Euclidean performance:
Euclid_metric = eye(size(samples_tst,2)); % Generate an identity matrix.
[mAP_Euclid_tst_set, prec_at_all_k_Euclid_tst_set, AUC_Euclid_tst_set] =...
    evaluate_precision(samples_tst, labels_tst, Euclid_metric);

fprintf('Euclidean: mean AP     tst_set = %f\n', mAP_Euclid_tst_set);
fprintf('Euclidean: AUC         tst_set = %f\n', AUC_Euclid_tst_set);
fprintf('Euclidean: precision@5 tst_set = %f\n', prec_at_all_k_Euclid_tst_set(5));


