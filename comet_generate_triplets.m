function [triplets_trn, triplets_trn_Q, triplets_trn_diff_mat]...
    = comet_generate_triplets( Tcnt, samples_trn, labels_trn, rnd_seed)
% [triplets_trn, triplets_trn_Q, triplets_trn_diff_mat] = ...
%     comet_generate_triplets( Tcnt, samples_trn, labels_trn, rnd_seed)
%
% Generate triplets for COMET training.
%
% input arguments:
% Tcnt               : Number of triplets to generate
% samples_trn        : samples x d features, matrix of training set samples
% labels_trn         : samples x 1           vector of training set labels
%                                            (numeric values). 
%                      ! labels should be in the range [1:n_classes] !
% n_classes          : Number of classes
% rndseed (optional) : seed to init triplets randomization with
%
% output arguments:
%
% triplets_trn         : triplets x 3           matrix triplets ids. 
%                        1st column is the query id, 2nd positive sample id
%                        , 3rd is negative sample id
% triplets_trn_Q       : triplets x d features, SPARSE matrix of query per 
%                        training triplet
% triplets_trn_diff_mat: triplets x d features, SPARSE matrix of difference
%                        between negative to positive samples per training
%                        triplet

%% init
if nargin >= 4
    % if given, init randomization with given seed
    rng('default')
    rng(rnd_seed);
end

N = size(samples_trn, 1);

% getting num of classes, and verifying labels are in the range [1:n_classes]
classes = unique(labels_trn).';
% n_classes = length(classes);
%assert(all(1:n_classes == sort(classes, 'ascend'))) ; % class labels should be in the range 1:n_classes, otherwise --> error
n_classes = max(classes);
assert(all(classes >=1)) ; % class labels should be in the range 1:n_classes, otherwise --> error

% find the ids of similar samples per class
[pos_samples_ids_per_class, neg_samples_ids_per_class] = deal(cell(n_classes,1));
for c = classes
    pos_samples_ids_per_class{c} = find(labels_trn == c);
    neg_samples_ids_per_class{c} = find(labels_trn ~= c);
end

%% generate the triplets
triplets_trn = zeros(Tcnt, 3);
for i=1:Tcnt
	query = randi(N);
	pos_smp = randsample(pos_samples_ids_per_class{labels_trn(query)},1);
	neg_smp = randsample(neg_samples_ids_per_class{labels_trn(query)},1);
	triplets_trn(i, :) = [query, pos_smp, neg_smp];
end

%% generate caching matrices

% generate a matrix that holds the Tcnt queries
triplets_trn_Q = samples_trn(triplets_trn(:,1), :); 

% generate a matrix that holds the difference between negative to positive 
% samples per training triplet
triplets_trn_diff_mat = samples_trn(triplets_trn(:,3), :) ...
    - samples_trn(triplets_trn(:,2), :);  


% make sure caching matrices are sparse
triplets_trn_Q = sparse(triplets_trn_Q);
triplets_trn_diff_mat = sparse(triplets_trn_diff_mat);



end

