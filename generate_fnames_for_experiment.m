function [ results_filename, model_filename ] =...
    generate_fnames_for_experiment(cfg_params, hyper_params, ofold, ifold)

% hyp_params is used for generating snapshot filenames (will make a filename
% out of this struct).

hyp_params.method = cfg_params.method_params_str; % Add method name to it.
hyp_params.dataset = cfg_params.dataset_params_str; % Add dataset name 

% Copy given hyper params to hyp_params struct.
hp_fields = fieldnames(hyper_params);
for i = 1:numel(hp_fields)
    hyp_params.(hp_fields{i}) = hyper_params.(hp_fields{i});
end

hyp_params.ofold = ofold;
hyp_params.ifold = ifold;

 [ results_filename, model_filename ] = ...
     generate_snapshot_fnames(hyp_params);

