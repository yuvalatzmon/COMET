function log_performance = ...
    comet_eval_performance ...
    (log_performance, curr_step, W, linlosses, cfg, cholLD)
% function log_performance 
%      = comet_eval_performance(log_performance, W, linlosses, cfg, cholLD)
%
% Evaluates performance of metric on test set.
%
% Input arguments:
% log_performance : A struct that contains performance log until this call.
% If it is empty then it is init here.
%
% curr_step      : The current step number.
% W              : The metric matrix to evaluate on.
% linlosses      : COMET training current linlosses.
% cfg
% cholLD         : LDLT Cholesky like decomposition (with CHOLMOD package).
%
% cfg.loss_type        : one of: 'hinge+logdet+frob', TBD
% cfg.logdet_weight    : Logdet regularizer coefficient
% cfg.sparse_weight    : Sparse regularizer coefficient
% cfg.log_len          : Total number of elements to log (used during init).
% cfg.samples_tst      : Test set samples to evaluate performance on (to be
%                        used when hfunc_eval_performance is used).
% cfg.labels_tst       : Test set labels to evaluate performance on (to be
%                        used when hfunc_eval_performance is used).
%
% Output arguments:
% log_performance  : A struct that contains performance log including this
%                   call.
%

% Check if this is the first evaluation, if yes, then call init procedure.
if isempty(fieldnames(log_performance))
    log_performance = init_log_performance(cfg.log_len, length(cfg.labels_tst));
end

t_tmp = datestr(now);

% Save the log count.
log_performance.log_cnt = log_performance.log_cnt+1;
log_cnt = log_performance.log_cnt;

% Save the current optimization step.
log_performance.logged_steps_nums(log_cnt) = curr_step;
% Save the trace of W.
log_performance.W_trace_history(log_cnt) = trace(W);

% Evaluate and save the train set loss, number of nnz on linlosses 
% triplets and get the Frob. norm.
[log_performance.train_set_losses(log_cnt), ...
    log_performance.train_set_losses_nnz(log_cnt), regularizer_norm, frobnorm] ...
    = evaluate_loss_over_train_set(cfg, W, linlosses, cholLD);
% Save the Frob. norm.
log_performance.W_frob_norm_history(log_cnt) = frobnorm;

% Save the regularizer norm.
log_performance.W_regularizer_norm_history(log_cnt) = regularizer_norm;

% Save the sparsity measure of the metric.
log_performance.W_sparsity_history(log_cnt) = nnz(W)/numel(W);

% Evaluate and save the test set mAP, prec@k, AUC 
[log_performance.tst_set_mean_avg_prec_history(log_cnt), ...
    log_performance.tst_set_prec_allk_history(:,log_cnt), ...
    log_performance.tst_set_AUC_history(log_cnt)] ...
    = evaluate_precision(cfg.samples_tst, cfg.labels_tst, W);

% Advance tstart by the time it took to evaluate the performance here.
log_performance.tend = datestr(now);
log_performance.tstart = datestr(datenum(log_performance.tstart) ...
    + datenum(log_performance.tend) - datenum(t_tmp)); 

if isfield(log_performance, 'iterlog')
    times_schur = 100*sum(log_performance.iterlog.used_schur_bound) ...
       /numel(log_performance.iterlog.used_schur_bound);

    d = size(W,1);
    x = (1:length(cumsum(log_performance.iterlog.is_zero_step == false)))/d;
    percent_active_steps = ...
        100*cumsum(log_performance.iterlog.is_zero_step == false)./(x*d);

else
    times_schur = nan;
    percent_active_steps = nan;
end


filtered_printf(cfg.show_progress_prints_flag,...
    ['prec@5 = %1.2f, mAP = %1.2f, AUC = %1.2f\n' ...
    'objective_loss = %10.2f\n' ...
    'Percent of active steps so far = %2.3f (%%)\n' ...
    'Percent used schur bound = %2.1f (%%)\n' ...
    'Elapsed training time = %s (HH:MM:SS)\n'], ...
    log_performance.tst_set_prec_allk_history(5,log_cnt), ...
    log_performance.tst_set_mean_avg_prec_history(log_cnt), ...
    log_performance.tst_set_AUC_history(log_cnt), ...
    log_performance.train_set_losses(log_cnt), ...
    percent_active_steps(end), times_schur, ...
    datestr(datenum(datestr(log_performance.tend)) ...
    - datenum(log_performance.tstart), 'HH:MM:SS'));

function  [loss, nnz_triplets_losses, regularizer_norm, frobnorm] ...
    = evaluate_loss_over_train_set(cfg, W, linlosses, cholLD)
% function  [loss, nnz_triplets_losses, frobnorm] ...
%     = evaluate_loss_over_train_set(cfg, W, linlosses, cholLD)
%
% Input arguments:
%
% cfg.loss_type        : one of: 'hinge+logdet+frob', TBD
% cfg.logdet_weight    : Logdet regularizer coefficient
% cfg.sparse_weight    : Sparse regularizer coefficient
% W              : The metric matrix to evaluate on.
% linlosses      : COMET training current linlosses.
% cholLD         : LDLT Cholesky like decomposition (with CHOLMOD package).
% 
%
% Output arguments:
%
% loss                : Total loss
% nnz_triplets_losses : Number of nonzero triplets hinge losses
% frobnorm            : Metric matrix frob norm.

frobnorm = norm(W, 'fro');
[~, D] = ldlsplit(cholLD);
if strcmp(cfg.loss_type,'hinge+logdet+L12')    
    Wregul = W;
    Wregul(eye(size(W))==1) = 0; % We exclude the main diagonal from L12.
    regularizer_norm = sum(sqrt(sum(Wregul.^2,2))); %L12;

    nz = (1+linlosses)>0;
    nnz_triplets_losses = nnz(nz);

    loss = sum(1+linlosses(nz)) - (cfg.logdet_weight)*sum(log(diag(D)))...
        + cfg.sparse_weight*regularizer_norm;
elseif strcmp(cfg.loss_type,'hinge+logdet')    
    regularizer_norm = nan;
    
    nz = (1+linlosses)>0;
    nnz_triplets_losses = nnz(nz);

    loss = sum(1+linlosses(nz)) - (cfg.logdet_weight)*sum(log(diag(D))) ; 
end

function log_performance = init_log_performance(log_len, n_samples_tst)
% Init the log_performance struct
log_performance.log_cnt = 0;
[log_performance.train_set_losses, ...
    log_performance.train_set_losses_nnz, ...
    log_performance.logged_steps_nums, ...
    log_performance.tst_set_mean_avg_prec_history, ...
    log_performance.tst_set_AUC_history, ...
    log_performance.W_frob_norm_history, ...
    log_performance.W_regularizer_norm_history, ...
    log_performance.W_sparsity_history, ...
    log_performance.W_trace_history] = deal(nan(log_len+1,1));
log_performance.tst_set_prec_allk_history= nan(n_samples_tst-1, log_len+1);
log_performance.tstart = datestr(now);
log_performance.tend = datestr(now);
