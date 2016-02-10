function [ W, cholLD, invW, linlosses] = diag_step0 ...
    (cfg, triplets_trn, triplets_trn_Q, triplets_trn_diff_mat)
% Make a first step. optimizing the main diagonal

d = size(triplets_trn_Q,2);


diag_W = ones(d,1);
diag_invW = ones(d,1); % Equals the log det gradient
linlosses = init_linlosses(triplets_trn, triplets_trn_Q, ...
    triplets_trn_diff_mat, diag_W);

QT_elements = triplets_trn_diff_mat.*triplets_trn_Q;
for n=1:cfg.n_steps_diag 
    [ diag_W, diag_invW, zero_step_ind] = single_diag_step ...
        (cfg, diag_W, diag_invW, linlosses, QT_elements);
    
    % update linlosses
    linlosses = init_linlosses(triplets_trn, triplets_trn_Q, ...
        triplets_trn_diff_mat, diag_W);
    
    if zero_step_ind
        break;
    end
end
W = sparse(1:d, 1:d, diag_W);
invW = sparse(1:d, 1:d, diag_invW);
cholLD = ldlchol(W);

function [ diag_W, diag_invW, zero_step_ind] = single_diag_step ...
    (cfg, diag_W, diag_invW, linlosses, QT_elements)
% function [ diag_W, diag_invW, zero_step_ind] = single_diag_step ...
%    (cfg, diag_W, diag_invW, linlosses, QT_elements)
%
% Performs a single graident step over a diagonal matrix.

CONDMAX = 1e6; % The max condition number (relation between maximal and minimal values
               % on the diagonal (eigen values)).

QT_IndsHingeLosses = bsxfun(@times, ((1+linlosses)>0), QT_elements);
diag_hinge_grad = sum(QT_IndsHingeLosses,1).';

minus_grad = (-1)*(diag_hinge_grad ...
    - cfg.logdet_weight*diag_invW); % Evaluate the step update.

% A do..while (like) loop, for protection against too big step sizes.
% It shouldn't run for many iterations since we have a logdet barrier.
stepsize_coeff = 1;
cnt = 0;
zero_step_ind = false;
while true
    update = diag_W + stepsize_coeff*cfg.stepsize_diag*minus_grad;
    condnum = max(update)/min(update); % The matrix updated condition num.
    if condnum < 0 || condnum > CONDMAX || any(update <0)
        stepsize_coeff = stepsize_coeff/2;
        cnt = cnt+1;
        if cnt == 20
            zero_step_ind = true;
            return;
        end
    else
        if cnt>1
            stepsize_coeff = stepsize_coeff/10;
        end
        break;
    end
        
end

diag_W = update;
diag_invW = update.^-1;



