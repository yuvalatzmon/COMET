function [samples_trn, samples_tst, labels_trn, labels_tst] = split_by_ratio(samples, labels, split_ratio)
% function [samples_trn, samples_tst, labels_trn, labels_tst] = split_by_ratio(samples, labels, split_ratio)
% split the dataset according to a split ratio
N = length(labels);
n_trn = floor(split_ratio*N);
trn_ids = randperm(N, n_trn) ;
tst_ids = 1:N;
tst_ids(trn_ids) = [];

samples_trn = samples(trn_ids, :);
samples_tst = samples(tst_ids, :);

labels_trn = labels(trn_ids);
labels_tst = labels(tst_ids);
end
