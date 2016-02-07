function [p,precision_at_all_ks] = get_avg_prec(a_ind)
% function [p,results_mean] = get_avg_prec(a_ind)
% a_ind is a binary matrix (or column vector).
% it is derived from the sorted scores of the classifier applied to the test set. It indicates whether the item at position i is a true positive.
% 
% so, a_ind(1)=1 if the top result is indeed a positive. a_ind(10)=0 if the tenth result wasn't a positive
% as in the other code, when sorting the scores to get a_ind, we should notice if the positives get the most negative or the most positive scores, and sort accordingly
% 
% output p is the average precision.
% output precision_at_all_ks is the vector of precisions at each of the k's


% set_size = sum(a_ind);
% if length(unique(set_size))~=1
%     warning('not all classes have same set size');
% else
%     set_size = unique(set_size);
% end

[N1,N] = size(a_ind);
if N1==1
    a_ind = a_ind';
    N = 1;
end

% evaluate 'precision at k' for all k's (number of positive indications (of higest k scores) / k)
results = cumsum(a_ind)./((1:size(a_ind,1))'*ones(1,size(a_ind,2)));

if nargout==2
    % mean over the queries
    precision_at_all_ks = mean(results,2);       
end

% mean over precision at k only at k's with positive indications
q = a_ind.*results;
p = zeros(1,N);
sa = sum(a_ind);
sanot0 = find(sa~=0);
p(sanot0) = sum(q(:,sanot0))./sa(sanot0); 

