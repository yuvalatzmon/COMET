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
%%% Off diagonal training hyper params
%
% num_d_repeats = Number of times repeating the iterations on the d coords.
% max_step_size = Maximal step size to take (step size can be smaller if  
%                                           bounded by the Schur complement).
% logdet_weight = Logdet regularizer coefficient.
% sparse_weight = Sparse regularizer coefficient.
% step_size_bound_weight = A multiplicative weight to stay off nulling
%                          of eigenvalues (stay off touching the PSD
%                          cone boundary), emprically chosen to be 0.3 .
%
%%% Main diagonal training hyper params:
%                          You can search these, hyper params independently of the off
%                          diagonal hyper params
% n_steps_diag = How may steps to take for training the elements of
%                          the main diagonal. 
% stepsize_diag = Diagonal optimization step size
% diag_L2_weight = Diagonal L2 regularization coefficient
%



%% Set parameters:

%dataset_name = 'protein';
dataset_name = 'RCV1_4_5K';

% Default values:
cfg.step_size_bound_weight = 0.3; % Default value is 0.3,
                                  % should fit most cases
cfg.stop_if_dense_above = 0.3; % Abort training if the density
                               % is above 30% on the first epoch

cfg.show_progress_prints_flag = true; % set to false supresses verbosity


if strcmp(dataset_name, 'protein')
    %%% Load and set the hyper params for the Protein dataset
    cfg.Tcnt = 20e3;
    cfg.seed = 0;
    cfg.stop_if_dense_above = 0.3;
    % Off diagonal training hyper params
    cfg.num_d_repeats = 48;
    cfg.max_step_size = 5e-4;
    cfg.sparse_weight = 0.012;
    cfg.logdet_weight = 5;
    % Main diagonal training hyper params
    cfg.diag_L2_weight = 0;
    cfg.stepsize_diag = 0; 
    cfg.n_steps_diag = 0;

    %% Load dataset
    [labels, samples ] = libsvmread('data/protein_libsvm_dataset');
    labels = labels + 1; % Labels must be in the range 1, 2, ... for the triplets generation


elseif strcmp(dataset_name, 'RCV1_4_5K')
    %%% Load and set the hyper params for the RCV1_4 with 5K features dataset
    cfg.Tcnt = 100e3;
    cfg.seed = 0;
    % Off diagonal training hyper params
    cfg.num_d_repeats = 12;
    cfg.max_step_size = .07;
    cfg.sparse_weight = 0.31;
    cfg.logdet_weight = 0.5;
    % Main diagonal training hyper params
    cfg.diag_L2_weight = 0;
    cfg.stepsize_diag = 10; 
    cfg.n_steps_diag = 30;

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


%% Set params for logging performance and snapshots during training. %%%%
% Between how many steps to log the performance.
d = size(samples_trn,2); % d is the number of features.
cfg.logging_steps_interval = d;
% A handle for a function to log the performance.
cfg.hfunc_eval_performance = @comet_eval_performance;
cfg.loss_type = 'hinge+logdet+L12';
cfg.samples_tst = samples_tst;
cfg.labels_tst = labels_tst;

%%% Set params for saving snapshot when performance logging is enabled. %%%
cfg.path_results_mat = './results_mat/';

% cfg.hyp_params is a struct used for generating snapshot filenames
% (will make a filename out of this struct fields).
cfg.hyp_params.method = 'COMETsparse+v1Prox';
cfg.hyp_params.dataset = dataset_name;
cfg.hyp_params.seed = cfg.seed;
cfg.hyp_params.Tcnt = cfg.Tcnt;
cfg.hyp_params.max_step_size = cfg.max_step_size;
cfg.hyp_params.logdet_weight = cfg.logdet_weight;
cfg.hyp_params.num_d_repeats = cfg.num_d_repeats;
cfg.hyp_params.sparse_weight = cfg.sparse_weight;

cfg.hyp_params.ofold = nan; % outer fold, when used in 2fold experiment
cfg.hyp_params.ifold = nan; % inner fold, when used in 2fold experiment


%% Generate triplets:
filtered_printf(cfg.show_progress_prints_flag, 'Generating a triplets set\n');
[ triplets_trn, triplets_trn_Q, triplets_trn_diff_mat ] = ...
    comet_generate_triplets( cfg.Tcnt, samples_trn, labels_trn, cfg.seed);
filtered_printf(cfg.show_progress_prints_flag, 'Done generating a triplets set\n');

cfg.triplets_trn = triplets_trn;


%% Train the metric learning algorithm:
[metric, LD, inv_metric, log_performance] =  train_sparse_comet(cfg, triplets_trn, triplets_trn_Q, ...
    triplets_trn_diff_mat);
[L, D] = ldlsplit(LD); % extract the L D components of the LDLT (Cholesky like) factorization
%% Evaluate precision @k performance:

fprintf('\n\nRESULTS:\n========\n');
% COMET performance:
log_cnt = log_performance.log_cnt;

fprintf('COMET    : mean AP     tst_set = %f\n', log_performance.tst_set_mean_avg_prec_history(log_cnt));
fprintf('COMET    : AUC         tst_set = %f\n', log_performance.tst_set_AUC_history(log_cnt));
fprintf('COMET    : precision@5 tst_set = %f\n', log_performance.tst_set_prec_allk_history(5, log_cnt));
fprintf('\n');

% Euclidean init performance:

fprintf('Euclidean: mean AP     tst_set = %f\n', log_performance.tst_set_mean_avg_prec_history(1));
fprintf('Euclidean: AUC         tst_set = %f\n', log_performance.tst_set_AUC_history(1));
fprintf('Euclidean: precision@5 tst_set = %f\n', log_performance.tst_set_prec_allk_history(5, 1));

fprintf('\nnnz(metric) %d, sparsity of metric = %1.3f\n', nnz(metric), nnz(metric)/numel(metric));


