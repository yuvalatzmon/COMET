function [W, LD] = ...
    train_dense_comet(cfg, triplets_trn, triplets_trn_Q, triplets_trn_diff_mat)
% [W, LD] = ...
%   train_dense_comet(cfg, triplets_trn, triplets_trn_Q, triplets_trn_diff_mat)
%
% Train dense COMET metric learning algorithm,
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
% cfg.frob_weight      : Frobenius regularizer coefficient
% cfg.seed             : Randomization seed
% cfg.show_prints_flag : Enables or supress training progress prints to
%                        screen.
%
% cfg.step_size_bound_weight : Weight in (0,1) to stay off nulling of 
%                              eigenvalues (empirically we set it to 0.1)
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
%                        




% init random number generator
rng('default')
rng(cfg.seed+1);

% init params
d = size(triplets_trn_Q, 2);
flgShow = cfg.show_progress_prints_flag;

% init state matrices & vectors
W  = eye(d);
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

%% training
filtered_printf(flgShow, 'Starting to train with dense COMET\n')
iter_num = 1; prev_progress = 0;

for k = coordinates_list
    
    [ W, LD, linlosses, step_size] = ...
        grad_stepk_dense(k, cfg, W, LD,  linlosses, ...
        triplets_trn_Q, triplets_trn_diff_mat);
  
    % display progress
    curr_progress = floor(100*(iter_num/length(coordinates_list) ));
    if curr_progress > prev_progress
        filtered_printf(flgShow, 'COMET: completed %d %%\n', curr_progress); 
    end
    prev_progress = curr_progress;
    
    % complete iteration
    iter_num = iter_num + 1;
end
%[L, D] = ldlsplit(LD);
%err = norm (L*D*L' - W, 1);
nnz(LD)

end

% function linlosses = init_linlosses(triplets_trn, triplets_trn_Q, ...
%     triplets_trn_diff_mat, W)
% % linlosses = init_linlosses(triplets_trn, triplets_trn_Q, ...
% %   triplets_trn_diff_mat, W)
% %
% % Evaluates the initial linlosses. linlosses is a vector that 
% % holds the (per triplet) value of the product of
% % query_smp.' * W * (neg_smp - pos_smp). Dimensions are triplets x 1
% %
% % It is named linlosses, because the loss per triplet 
% % equals [1 + linlosses]_+

% Tcnt = size(triplets_trn_Q,1);
% linlosses = zeros(Tcnt,1);
% samples_num = max(triplets_trn(:,1));
% triplets_trn_diff_mat_T = triplets_trn_diff_mat.'; % selecting columns is much faster with sparse matrices
% for n=1:samples_num
%     t_q_eq_n_condition = (triplets_trn(:,1) == n);
%     query_smp = triplets_trn_Q(find(t_q_eq_n_condition,1),:);
%     q_W = sparse(query_smp*W);
%     linlosses(t_q_eq_n_condition) = ...
%         (q_W*triplets_trn_diff_mat_T(:, t_q_eq_n_condition));
    
% end
% end
