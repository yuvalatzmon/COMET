function linlosses = init_linlosses(triplets_trn, triplets_trn_Q, ...
    triplets_trn_diff_mat, diag_W)
% linlosses = init_linlosses(triplets_trn, triplets_trn_Q, ...
%   triplets_trn_diff_mat, W)
%
% Evaluates the initial linlosses. linlosses is a vector that
% holds the (per triplet) value of the product of
% query_smp.' * W * (neg_smp - pos_smp). Dimensions are triplets x 1
%
% It is named linlosses, because the loss per triplet
% equals [1 + linlosses]_+
%
% Input variables:
% triplets_trn             : List of triplets
% triplets_trn_Q           : Each row is a query samples of each triplet
% triplets_trn_diff_mat    : Each row equals (p^- - p+) of each triplet
% diag_W                   : *diagonal* elements of W
% 
% 


Tcnt = size(triplets_trn_Q,1);
linlosses = zeros(Tcnt,1);
samples_num = max(triplets_trn(:,1));
triplets_trn_diff_mat_T = triplets_trn_diff_mat.'; % selecting columns is
                                                   % much faster with 
                                                   % sparse matrices
for n=1:samples_num
    t_q_eq_n_condition = (triplets_trn(:,1) == n);
    q_id = find(t_q_eq_n_condition,1);
    if isempty(q_id)
        continue;
    end
    query_smp = triplets_trn_Q(q_id,:);
    q_W = sparse(query_smp.*diag_W(:).');
    linlosses(t_q_eq_n_condition) = ...
        (q_W*triplets_trn_diff_mat_T(:, t_q_eq_n_condition));
    
end
end
