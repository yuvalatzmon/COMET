function [mean_avg_precision, precision_at_all_ks, auc] = evaluate_precision(samples, labels, metric_matrix)
% function [mean_avg_precision, precision_at_all_ks] = evaluate_precision(samples, labels, W)
% Evaluate mean avg precision,  precision_at_all_ks and AUC with a given similarity matrix

all_images_similiarities = transpose(samples*metric_matrix*samples.'); % evlaute similiarity scores between all images
auc_score_per_example = zeros(size(samples,1),1);

ranking_inds = zeros(size(all_images_similiarities) - [1 0]); % this will hold the ranking of positive images per image:label
for n=1:size(samples,1) % iterating over each of the images
    tmp_ranking = [all_images_similiarities(:,n) , labels == labels(n)]; % taking the similarity of image 'n' to all other images
    tmp_ranking(n,:) = []; % taking out the similarity of an image to itself
    tmp_ranking = sortrows(tmp_ranking, -1); % ranking according to the similiarity score
    ranking_inds(:,n) = tmp_ranking(:,2); % extracting the ranking of positive images per image:label
    
    auc_score_per_example(n) = scoreAUC(tmp_ranking(:,2)==1, tmp_ranking(:,1));
end

[avg_precision, precision_at_all_ks] = get_avg_prec(ranking_inds); % evaluate the mean avg precision
mean_avg_precision = mean(avg_precision);
auc = mean(auc_score_per_example);
end
