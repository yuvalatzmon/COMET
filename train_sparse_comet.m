function [W, LD, invW, log_performance] = ...
    train_sparse_comet(cfg, triplets_trn, triplets_trn_Q, triplets_trn_diff_mat)
% [W, LD, invW] = ...
%   train_sparse_comet(cfg, triplets_trn, triplets_trn_Q, triplets_trn_diff_mat)
%
% Train COMET metric learning algorithm, with a proximal sparse step.
%                                         ("Metric Learning One Feature at a
%                                         Time", Y. Atzmon, U. Shalit,
%                                         G. Chechik, 2015 )
%
% Input arguments:
% cfg.num_d_repeats    : Number of times repeating the iterations on the d
%                        coordinates.
% cfg.max_step_size    : Maximal step size to take (step size can be
%                        smaller if bounded by the Schur complement.)
% cfg.logdet_weight    : Logdet regularizer coefficient
% cfg.sparse_weight    : Sparse regularizer coefficient
% cfg.seed             : Randomization seed
% cfg.show_prints_flag : Enables or supress training progress prints to
%                        screen.
%
% cfg.step_size_bound_weight : Weight in (0, 1) to stay off nulling of
%                              eigenvalues (empirically we set it to 0.1)
%
% cfg.logging_steps_interval : Between how many steps to log the
%                              performance. Can be set to NaN.
% cfg.hfunc_eval_performance : a handle for a function to log the
%                              performance. Can be set to NaN.
% cfg.samples_tst      : Test set samples to evaluate performance on (to be
%                        used when hfunc_eval_performance is used).
% cfg.labels_tst       : Test set labels to evaluate performance on (to be
%                        used when hfunc_eval_performance is used).
%
% cfg.diag_L2_weight   : Weight coefficient for L2 regularization of the
%                        diagonal
%
% triplets_trn         : triplets x 3           matrix triplets ids.
%                        1st column is the query id, 2nd positive sample id
%                        , 3rd is negative sample id
% triplets_trn_Q       : triplets x d features, SPARSE matrix of query per
%                        training triplet
% triplets_trn_diff_mat: triplets x d features, SPARSE matrix of difference
%                        between negative to positive samples per training
%                        triplet.
%
% Output arguments:
%
% W    : The resulting metric matrix.
% LD   : Resulting Cholsky LDL decomposition, according to CHOLMOD package.
% invW : Inverse of W
% log_performance : training performance log. Used if logging is configured
% at cfg struct




%% init


% init random number generator
rng('default')
rng(cfg.seed+1);

% init params
d = size(triplets_trn_Q, 2);
flgShow = cfg.show_progress_prints_flag;
log_performance = struct();

% init state matrices & vectors
W  = eye(d);
invW = eye(d);
if is_pattern_in_str('Prox', cfg.hyp_params.method)
    % Init non overlapping groups 1:d (each is a column). Group 0 is
    % optimized directly on the main diagonal of W;
    non_overlap_V = sparse(d-1, d); 
else
    non_overlap_V = [];
end

LD = ldlchol(sparse(W)); % LDL decomposition, takes only sparse matrices

filtered_printf(flgShow, 'Evaluates the initial linlosses\n');
t_linlosses = tic;
linlosses = init_linlosses(triplets_trn, triplets_trn_Q, triplets_trn_diff_mat, diag(W));
%load protein_init_linlosses
%fprintf('Warning, loaded from file protein_init_linlosses');
if flgShow; toc(t_linlosses); end

% generate features (coordinates) list to iterate upon
coordinates_list = [];
for k=1:cfg.num_d_repeats
    coordinates_list = [coordinates_list randperm(d)];
end
coordinates_list = coordinates_list(:).';

% init iterations logger (to remove in publication version)
nsteps = length(coordinates_list);
if is_pattern_in_str('L12', cfg.hyp_params.method)
    iterlog = struct();
elseif is_pattern_in_str('v1Prox', cfg.hyp_params.method)
    [iterlog.is_zero_step, ...
        iterlog.is_origin_PD, ...
        iterlog.used_schur_bound,...
        iterlog.actual_step_size] = deal([]);
end

%% training
ST = dbstack;
func_name = ST.name;
filtered_printf(flgShow, 'Starting to train with %s\n', func_name)
tic_train = tic;
iter_num = 1; prev_progress = 0;
cfg_d_repeats = cfg.num_d_repeats;

if cfg_d_repeats == 0 % this setup will only train the main
                      % diagonal
    coordinates_list = -1; % indicates that we only train the main diagonal
end
for k = coordinates_list
        
        
    % If logging is configured (slows down training), then evaluate
    % performance every cfg.logging_steps_interval steps
    log_performance = do_logging_if_required ...
        (log_performance, iter_num, cfg, W, invW, LD, linlosses, non_overlap_V);

    % In case we do prox steps, then we take a first initial step of
    % optimizing the main diagonal.
    if iter_num == 1 && is_pattern_in_str('v1Prox', cfg.hyp_params.method)
        % if main diag train results are cached, then load cached results.
        do_force = 0;
        backup_hp = cfg.hyp_params;
        cfg.hyp_params.num_d_repeats = 0;
        cfg.hyp_params.diag = '';
        [ ~, model_filename] = ...
            generate_snapshot_fnames(cfg.hyp_params);        
        full_model_fname = fullfile(cfg.path_results_mat, model_filename);
        performance_results_vars = {'trained_obj'};
        [do_calc_results,  trained_obj] = ...
            cond_load([full_model_fname '.mat'], do_force, ...
            performance_results_vars{1:end});        
        if do_calc_results
            [ W, LD, invW, linlosses] = ...
                diag_step0(cfg, ...
                triplets_trn, triplets_trn_Q, triplets_trn_diff_mat);
            % If logging is configured (slows down training), then evaluate
            % performance every cfg.logging_steps_interval steps
            log_performance = do_logging_if_required ...
                (log_performance, iter_num, cfg, W, invW, LD, linlosses, non_overlap_V);
        else
            W = trained_obj.metric;
            LD = trained_obj.LD;
            invW = trained_obj.invW;
            linlosses = trained_obj.linlosses;
            log_performance = trained_obj.log_performance;

            % adjust timing to current time
            tend = datestr(now);
            log_performance.tstart = ...
                datestr(datenum(tend) - ...
                (datenum(log_performance.tend) - ...
                 datenum(log_performance.tstart) ) ...
                );
            log_performance.tend = tend;
            filtered_printf(flgShow,'Loaded cached diagonal training.\n');
        end
        cfg.hyp_params = backup_hp;
        
        iter_num = iter_num +  cfg.n_steps_diag;
    end
    
    if k == -1
        break; % break if we only train the main diagonal
    end
    % Do a single coordinate step:
    
    if is_pattern_in_str('L12', cfg.hyp_params.method)
        % Do L12 regularization step
        [ W, LD, invW, linlosses, step_size] = ...
            grad_stepk_sparse(k, cfg, W, LD, invW, linlosses, ...
            triplets_trn_Q, triplets_trn_diff_mat);
    elseif is_pattern_in_str('v1Prox', cfg.hyp_params.method)
        % Do a proximal step (version 1)
        [ W, LD, invW, non_overlap_V, linlosses, step_size, iterlog] = ...
            prox1_stepk(k, cfg, W, LD, invW, non_overlap_V, ...
            linlosses, triplets_trn_Q, triplets_trn_diff_mat, iterlog);        
    else
        error('unknown method %s\n', cfg.hyp_params.method);
    end
    
    if iter_num <=d && mod(iter_num, ceil(d/5)) == 0
        currVsparsity = nnz(non_overlap_V(:, coordinates_list(1:iter_num)))...
            /numel(non_overlap_V(:, coordinates_list(1:iter_num)));
        filtered_printf(flgShow, ...
            'sparsity of V on touched coords= %1.3f, sqrt(d)/d = %1.3f\n', ...
            currVsparsity ,sqrt(d)/d );
        if currVsparsity > cfg.stop_if_dense_above
            % if matrix is too dense then
            % killing this job by breaking the training, but first generate a false results file so
            % no other process will try to repeat this job, by advancing
            % the iter num to the last iteration
            iter_num = 1+length(coordinates_list) + cfg.n_steps_diag;
            non_overlap_V = -1 + zeros(d-1, d); % override all Vs (and make it dense) to mark that we've break this traininig.
            cfg.num_d_repeats = cfg_d_repeats;

            filtered_printf(flgShow, '\n ! ! ! Matrix is TOO DENSE! BREAKING the training ! ! !\n\n');
            break;
            
        end
    end
    log_performance.iterlog = iterlog; %to remove on publication ver.

    % display progress
    curr_progress = floor(100*(iter_num/length(coordinates_list) ));
    if curr_progress > prev_progress
        filtered_printf(flgShow, 'COMET: completed %d %%\n', curr_progress);
    end
    prev_progress = curr_progress;
    
    % complete iteration
    iter_num = iter_num + 1;
end
if flgShow; toc(tic_train); end

% If logging is configured (slows down training), then evaluate
% performance after the last iteration
log_performance =...
    do_logging_if_required ...
    (log_performance, iter_num, cfg, W, invW, LD, linlosses, non_overlap_V);
end


function log_performance = do_logging_if_required ...
    (log_performance, step_num, cfg, W, invW, LD, linlosses, non_overlap_V)
% Check if logging is required. If yes,
% log every cfg.logging_steps_interval
if ~isnan(cfg.logging_steps_interval) ...
        && isa(cfg.hfunc_eval_performance, 'function_handle')
    
    
    d = size(W,1); % dimension
    
    % Check if logging is required. If yes,
    % log every cfg.logging_steps_interval or after the last itertion
    if cfg.n_steps_diag>0 && (step_num-1) == cfg.n_steps_diag
        cfg.hyp_params.diag = '';
    end
    
    if (step_num-1) == 0 ||...
       (mod((step_num-1), cfg.logging_steps_interval)==cfg.n_steps_diag);
   
        filtered_printf(cfg.show_progress_prints_flag, ...
            'Evaluating performance (for logging) @ step #%d\n'...
            , step_num-1);
       
        % Number of elements to log.
        cfg.log_len = 1+ceil(cfg.num_d_repeats*d/cfg.logging_steps_interval);
        
        log_performance = ...
            cfg.hfunc_eval_performance ...
            (log_performance, step_num-1, W, linlosses, cfg, LD);

        % Evaluate the row sparsity of V (with proximal updates). 
        if is_pattern_in_str('v1Prox', cfg.hyp_params.method)
            
            if ~isfield(log_performance, 'V_row_sparsity'); 
                log_performance.V_row_sparsity = []; 
            end
            
            % we actually work in columns because matlab is efficient for 
            % column sparsity.
            nz_rows = sum(non_overlap_V ~= 0, 1);
            log_performance.V_row_sparsity(end+1) = ...
                nnz(nz_rows)/numel(nz_rows);
            filtered_printf(cfg.show_progress_prints_flag,...
                '\n Row-Column sparsity of V = %1.3f\n', ...
                log_performance.V_row_sparsity(end));
            
        end
        
        log_performance = snapshot_model_and_results ...
            (log_performance, step_num, cfg, W, invW, non_overlap_V, LD, linlosses);
        
    end
    if cfg.n_steps_diag>0 && (step_num-1) == cfg.n_steps_diag
        rmfield( cfg.hyp_params, 'diag');
    end
    
end
end


function log_performance = snapshot_model_and_results ...
    (log_performance, step_num, trained_obj, W, invW, non_overlap_V, LD, linlosses)

t_tmp = datestr(now);
d = size(W,1); % dimension
current_d_step = floor((step_num-1)/ d);

if isfield(trained_obj, 'git_commit_hash')
    git_commit_hash = trained_obj.git_commit_hash;
else
    git_commit_hash = '';
    trained_obj.git_commit_hash = git_commit_hash;
end

trained_obj.hyp_params.num_d_repeats = current_d_step;
[ results_filename, model_filename] = ...
    generate_snapshot_fnames(trained_obj.hyp_params);

trained_obj.log_performance = log_performance;
trained_obj.step_num = step_num;
trained_obj.metric = W;
trained_obj.invW = invW;
trained_obj.V = non_overlap_V;
trained_obj.LD = LD;
trained_obj.linlosses = linlosses;
trained_obj.log_performance = log_performance;

[tstart, tend] = deal(log_performance.tstart, log_performance.tend);
save('-v7.3', fullfile(trained_obj.path_results_mat, model_filename), ...
    'git_commit_hash', 'trained_obj', 'tstart', 'tend');
fprintf('saved model snapshot: %s.mat\n',fullfile(trained_obj.path_results_mat, model_filename));

snapshot_results(trained_obj, results_filename);

clear trained_obj

% Advance tstart by the time it took to save the results files.
log_performance.tend = datestr(now);
log_performance.tstart = datestr(datenum(log_performance.tstart) ...
    + datenum(log_performance.tend) - datenum(t_tmp)); 

end

function snapshot_results(trained_obj, results_filename)
log_perf = trained_obj.log_performance;
git_commit_hash = trained_obj.git_commit_hash;
tstart = trained_obj.log_performance.tstart;
tend = trained_obj.log_performance.tend;

% do evaluation on weak supervision data/algo results
tstart_eval_prec = datestr(now); % redundant - for backward compatability 
tend_eval_prec = datestr(now); % redundant - for backward compatability 

log_cnt = log_perf.log_cnt;
tst_set_prec_allk = log_perf.tst_set_prec_allk_history(:, log_cnt);
tst_set_mean_avg_prec = log_perf.tst_set_mean_avg_prec_history(log_cnt);
tst_set_AUC = log_perf.tst_set_AUC_history(log_cnt);

d = size(trained_obj.metric,1);
result_criteria.prec1  = tst_set_prec_allk(1);
result_criteria.prec5  = tst_set_prec_allk(5);
result_criteria.prec10 = tst_set_prec_allk(10);
result_criteria.mAP    = tst_set_mean_avg_prec;
result_criteria.AUC    = tst_set_AUC;
result_criteria.V_row_sparsity = log_perf.V_row_sparsity(log_cnt);

% The below criterion is used to do a first iteration of single epoch
% selection of hyper of params that give a sparsity of ~O(sqrt(d)). 
target_sparsity = mean([sqrt(d), sqrt(d)*log10(d) ])/d; 
result_criteria.minus_diff_Vsparsity_from_Osqrtd = ...
    1 - abs(log_perf.V_row_sparsity(log_cnt) - target_sparsity);


performance_results_vars = {'tst_set_prec_allk', 'tst_set_mean_avg_prec',...
    'tst_set_AUC', 'result_criteria', 'tstart', 'tend', ...
    'tstart_eval_prec', 'tend_eval_prec'};

%% save the training performace results + the current git commit hash
try
    performance_results_vars{end+1} = 'git_commit_hash';
    
    full_fname_results = ...
        fullfile(trained_obj.path_results_mat, results_filename);
    save(full_fname_results, performance_results_vars{1:end})

    fprintf('saved results snapshot: %s.mat\n',full_fname_results);
catch e
    disp(getReport(e, 'extended'))
    clk = clock;
    disp(num2str(clk));
    screenSize = get(0,'ScreenSize');
    if ~isequal(screenSize(3:4),[1 1])
        keyboard
    end
end

end
