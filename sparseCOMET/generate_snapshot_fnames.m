function [ results_filename, model_filename ] = generate_snapshot_fnames(hyp_params)

results_filename = regexprep(sprintf('results_%s', buildStringFromStruct(hyp_params, '__')), '\.', '_' );
model_filename = regexprep(sprintf('model_%s', buildStringFromStruct(hyp_params, '__')), '\.', '_' );
