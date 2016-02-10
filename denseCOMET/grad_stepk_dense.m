function [ new_W, new_cholLD, linlosses, eta, myRcond]= ...
    grad_stepk_dense(k, cfg, W, cholLD,  linlosses, triplets_trn_Q, ...
    triplets_trn_diff_mat)
% [ new_W, new_cholLD, linlosses, step_size_bound, myRcond]= ...
%     grad_stepk_dense(k, cfg, W, cholLD,  linlosses, triplets_trn_Q, ...
%     triplets_trn_diff_mat)
% A gradient step on a feature (a row-column coordinate k), with embedding
%
% Input arguments:
% k         : The feature id on the metric matrix.
% cfg       : See train_comet.m
% W         : The current metric matrix.
% linlosses : A vector that holds the value of the product of 
%             query_smp.' * W * (neg_smp - pos_smp), per triplet. 
%             Dimensions are triplets x 1. It is named linlosses, 
%             because the loss per triplet equals [1 + linlosses]_+
%
% triplets_trn_Q        : See train_comet.m
% triplets_trn_diff_mat : See train_comet.m
% cholLD    : LD of Cholesky decomposition (of CHOLMOD package)
%
% output arguments:
% new_W           : The newly evaluated metric matrix.
% new_cholLD      : updated LD of Cholesky decomposition (of CHOLMOD package)
% linlosses       : The newly evaluated linlosses.
% eta             : The current step size bound (for logging) .
% myRcond         : new_W matrix condition number (evaluated as in matlabs 
%                   rcond) .

%% init locals
d = size(W,1);

%% Eval the raw gradient for the row k: k,l: sum: Q(:,k).*T(:,l).*indicator(1+linlosess(:))
Qk_m_HingeLosses = triplets_trn_Q(:,k).*((1+linlosses)>0);
u_QmT = (Qk_m_HingeLosses.'*triplets_trn_diff_mat).';
%% Eval the raw gradient for the column k: l,k
Tk_m_HingeLosses = triplets_trn_diff_mat(:,k).*((1+linlosses)>0);
u_TmQ = (Tk_m_HingeLosses.'*triplets_trn_Q).';

%% Eval the gradient symmetric update:
e_k = sparse(zeros(d, 1)); e_k(k) = 1;
d_logdet = ldlsolve(cholLD, e_k); % equals A^-1(:,k)

u = (-1)*((u_QmT+u_TmQ)/2 + ...
    cfg.frob_weight*W(:,k) - cfg.logdet_weight*(d_logdet));

%% EVAL A^-1 
ids_A = true(1,d);
ids_A(k) = false;

%% EVAL A^-1*B


% Delete row/col k of W by the Cholesky decompopsition
% Note that deleting a row/col actually set it to the  kth row/col of identity
LD_A = ldlrowmod(cholLD, k); 

B = sparse(W(:, k)); B(k) = 0;
invA_B = ldlsolve(LD_A,  B); invA_B(k) = [];

%% Eval step size bound:

% Calculates the quadratic eq. coefficients.
% a = u(ids_A).'*invA*u(ids_A);

u_A = sparse(u); u_A(k) = 0;
invA_u = ldlsolve(LD_A,  u_A); invA_u(k) = [];
a = u(ids_A).'*invA_u;

b = -2*(u(k) - u(ids_A).'*invA_B);

c = -(W(k,k) -  W(k, ids_A)*invA_B);

eta_roots = roots([a b c]); % A bound for the step size (eta)
if any(imag(eta_roots(:))) % If roots are imaginary then step size = 0
    schurcond_bound_grad = 0;
    if a>0
        error(['eta roots is complex and a > 0 --> i.e. inequiality' ...
            ' ax^2+bx+c<0 doesn''t hold!!'])
    end
elseif a == 0 && b ==0 % If a,b  == 0 then ineq always holds for c<0 and for all eta.
    if c<0
        schurcond_bound_grad = inf;
    else
        error('inequality: (%f)*eta^2 + (%f)*eta + (%f)\n < 0 is false',...
            a, b, c);
    end
    
elseif a ==0
    if b<0 && c<0
        schurcond_bound_grad = inf;
    elseif b>0 && c<0
        schurcond_bound_grad = -c/b;
    else
        error('inequality: (%f)*eta^2 + (%f)*eta + (%f)\n < 0 is false',...
            a, b, c); 
    end
    
else
    if max(eta_roots)<0 % Adding some tolerance to numerical instabilities 
                        % --> if both roots are negative, take a 0 step.
        fprintf('\n\nmax(eta_roots)<0 and equals %.1e\n\n', max(eta_roots));
        eta_roots(2) = 0;
    end
    
    % Weight to stay off nulling of eigenvalues.
%     step_size_bound = cfg.step_size_bound_weight*max(eta_roots); 

    schurcond_bound_grad = max(eta_roots);

end
% eta = min(cfg.max_step_size, step_size_bound-eps); % Update step size.

% Update step size.
eta = min([cfg.max_step_size, ...
    schurcond_bound_grad*cfg.step_size_bound_weight]); 

%%%%%%%%%%%%%%%%%%%%%%%%%

%% Update W
new_W = W;
new_W(:,k) = W(:,k) + eta*u;
new_W(k,:) = new_W(k,:) + eta*u.';

% Update embedding (LDL Cholesky decomposition)
% Note that there is no support for change row/col, so in order to change
% a row/col we first need to delete it, and then add it back with the
% updated values. Also note that deleting a row/col actually set it to the 
% kth row/col of identity, and that adding a row assumes kth row/col is a kth
% row/col of identity.
updated_row = sparse(new_W(:,k));
new_cholLD = ldlrowmod (LD_A, k, updated_row) ;%add row


%% Eval the linear losses for the row and column k:
QmT_u = triplets_trn_Q*(eta*u);
QmT_u = triplets_trn_diff_mat(:,k).*QmT_u;

TmQ_u = triplets_trn_diff_mat*(eta*u);
TmQ_u = triplets_trn_Q(:,k).*TmQ_u;
linlosses = linlosses + QmT_u + TmQ_u;
 

end

