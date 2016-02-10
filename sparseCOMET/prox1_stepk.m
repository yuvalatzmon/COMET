function [ new_W, new_cholLD, new_invW, new_V, linlosses, step_size_bound_grad ]= ...
    prox1_stepk(k, cfg, W, cholLD,  invW, V, linlosses, triplets_trn_Q, ...
    triplets_trn_diff_mat)
% function [ new_W, new_cholLD, new_invW, new_V, linlosses, step_size_bound_grad ]= ...
%     prox1_stepk(k, cfg, W, cholLD,  invW, V, linlosses, triplets_trn_Q, ...
%     triplets_trn_diff_mat)
%
% A proximal (sparse) step on a feature (a row-column coordinate k)
%
% Input arguments:
% k         : The feature id on the metric matrix.
% cfg       : See train_sparse_comet.m
% W         : The current metric matrix.
% cholLD    : LD of LDLT decomposition (of CHOLMOD package)
% invW      : The inverse of the current metric matrix.
% V         : Non overlapping terms of the metric.
% linlosses : A vector that holds the value of the product of
%             query_smp.' * W * (neg_smp - pos_smp), per triplet.
%             Dimensions are triplets x 1. It is named linlosses,
%             because the loss per triplet equals [1 + linlosses]_+
%
% triplets_trn_Q        : See train_comet.m
% triplets_trn_diff_mat : See train_comet.m
%
% output arguments:
% new_W           : The newly evaluated metric matrix.
% new_cholLD      : Updated LD of LDLT decomposition (of CHOLMOD package)
% new_invW        : The inverse of the newly evaluated metric matrix.
% new_V           : Updated non overlapping terms of the metric.
% linlosses       : The newly evaluated linlosses.
% step_size_bound_grad : The current step size bound (for logging) .

%% init locals
d = size(W,1);

%% Eval the raw gradient for the row k: k,l: sum: Q(:,k).*T(:,l).*indicator(1+linlosess(:))
Qk_m_HingeLosses = triplets_trn_Q(:,k).*((1+linlosses)>0);
u_QmT = (Qk_m_HingeLosses.'*triplets_trn_diff_mat).';
%% Eval the raw gradient for the column k: l,k
Tk_m_HingeLosses = triplets_trn_diff_mat(:,k).*((1+linlosses)>0);
u_TmQ = (Tk_m_HingeLosses.'*triplets_trn_Q).';

%% Eval the gradient symmetric update:
d_logdet = invW(k,:).';

minus_grad = (-1)*((u_QmT+u_TmQ)/2 + ...
    - cfg.logdet_weight*(d_logdet));
minus_grad(k) = []; %remove the main diagonal element;

%% Make a proximal update, with the step_size given by hyper params

hp_stepsize = cfg.max_step_size;
new_v = prox_updatek(k, hp_stepsize*cfg.step_size_bound_weight, ...
    cfg.sparse_weight, V, minus_grad);

% If the update is zero. Then return.
if all(new_v == V(:,k))
    new_W = W;
    new_cholLD = cholLD;
    new_invW = invW;
    new_V = V;
    step_size_bound_grad = inf;
    return;    

end

%% EVAL A^-1
ids_A = true(1,d);
ids_A(k) = false;

%% EVAL A^-1*B

% Delete row/col k of W by the Cholesky decompopsition
% Note that deleting a row/col actually set it to the  kth row/col of
% identity. We overcome this by setting next 'B(k) = 0'
LD_A = ldlrowmod(cholLD, k);

B = sparse(W(:, k)); B(k) = 0;
invA_B = ldlsolve(LD_A,  B); invA_B(k) = [];

%% Eval step size bound:

minus_grad_with_zero_main_diag = insert_to_vec_at_pos_k(0, k, minus_grad);

[schurcond_bound_grad] = ...
    eval_maxstep_size_with_schur_comp...
    (k, W, minus_grad_with_zero_main_diag, ids_A, LD_A, invA_B);

[step_size_bound_grad, is_schurcond_bound] ...
    = min([cfg.max_step_size, schurcond_bound_grad]); % Update step size.
is_schurcond_bound = is_schurcond_bound-1;




old_v_with_main_diag_element = sparse(insert_to_vec_at_pos_k(0, k, V(:,k)));

if norm(old_v_with_main_diag_element) > 0
    origin_direction = ...
        -old_v_with_main_diag_element/ norm(old_v_with_main_diag_element);
    
    [step_size_bound_origin] = ...
        eval_maxstep_size_with_schur_comp...
        (k, W, origin_direction, ids_A, LD_A, invA_B);
    
    is_origin_PD = norm(old_v_with_main_diag_element) < step_size_bound_origin;

    if is_schurcond_bound
        backoff_weight = cfg.step_size_bound_weight;
    else
        backoff_weight = 1;
    end

else
    % A coefficient to stay off PSD.
    if is_schurcond_bound
        backoff_weight = cfg.step_size_bound_weight;
    else
        backoff_weight = 1;
    end
    is_origin_PD = true;
end

if is_origin_PD 
    % if is_origin_PD ,
    % then any prox step, with a eta_grad step size, keeps PSD.
    
    % If cfg.sparse_weight==0 then we need to take a back-off from the PSD
    % cone boundary. Because otherwise, a PSD solution will be ill-conditioned.
    if cfg.sparse_weight == 0
        error('Zero sparse weight is not supported. Use dense COMET instead.');
    end
       
    prox_step_size = backoff_weight*step_size_bound_grad;
    new_v = prox_updatek(k, prox_step_size, cfg.sparse_weight, V, minus_grad);
 
else
    warning('Skipped update, origin is not PD\n');
    new_W = W;
    new_cholLD = cholLD;
    new_invW = invW;
    new_V = V;
    step_size_bound_grad = inf;
    return;    
end    
   


%% Update W, V

new_V = V;
new_V(:,k) = new_v;

new_W = W;

new_v_with_main_diag_element = insert_to_vec_at_pos_k(0, k, new_v);

new_W(:,k) = ...
  new_v_with_main_diag_element + W(:,k) - old_v_with_main_diag_element;
new_W(k,:) = ...
  new_v_with_main_diag_element.' + W(k,:) - old_v_with_main_diag_element.';

update = new_W(:,k) - W(:,k);



%% update W^-1
e_k = zeros(d, 1); e_k(k) = 1;
U = [update, e_k];
V = [e_k.';update.'];

%%%% new_invW = invW - invW*U*(eta^-1*eye(2)+V*invW*U)^(-1)*V*invW; %%%%
VW1 = V*invW;
W1U = invW*U;
VW1U = V*W1U;
C1 = (eye(2)+VW1U)^(-1);
W1update_rhs = C1*VW1;
W1update = W1U*W1update_rhs;
new_invW = invW - W1update;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Update embedding (LDL Cholesky decomposition)
% Note that there is no support for change row/col in CHOLMOD, so in order to change
% a row/col we first need to delete it, and then add it back with the
% updated values. Also note that deleting a row/col actually set it to the
% kth row/col of identity, and that adding a row assumes kth row/col is a kth
% row/col of identity.
updated_row = sparse(new_W(:,k));
new_cholLD = ldlrowmod (LD_A, k, updated_row) ;%add row


%% Eval the linear losses for the row and column k:
QmT_u = triplets_trn_Q*(update);
QmT_u = triplets_trn_diff_mat(:,k).*QmT_u;

TmQ_u = triplets_trn_diff_mat*(update);
TmQ_u = triplets_trn_Q(:,k).*TmQ_u;
linlosses = linlosses + QmT_u + TmQ_u;
end


function vec_new = insert_to_vec_at_pos_k(val, k, vec)
% function vec_new = insert_to_vec_at_pos_k(val, k, vec)
% Inserts (appends) a value at position k.
% Note that it always returns a column vector.

vec_new = nan(length(vec)+1,1);
vec_new(1:(k-1)) = vec(1:(k-1));
vec_new(k) = val;
vec_new((k+1):end) = vec(k:end);
end

function z = prox_updatek(k, stepsize, sparse_weight, V, minus_grad)
% function z = prox_updatek(k, stepsize, sparse_weight, V, minus_grad)
% z = h * [1 - sparse_weight/norm(h)]_+
% Where: h = vk - stepsize * gradient
% This evaluation costs O(d).

h = V(:,k) + stepsize*minus_grad; 

% sparse_weight
% norm(V(:,k))
% norm(h)
if norm(h) == 0
    z = 0;
else
    x = 1-sparse_weight/norm(h);
%     x % PRINTING TO SCREEN
    x = x.*(x>0);
    
    z = h*x;
end
end

function [step_size_bound] = eval_maxstep_size_with_schur_comp ...
    (k, W, u, ids_A, LD_A, invA_B)
% function [step_size_bound] = eval_maxstep_size_with_schur_comp ...
%     (k, W, u, ids_A, LD_A, invA_B)

u_A = sparse(u); u_A(k) = 0;
invA_u = ldlsolve(LD_A,  u_A); invA_u(k) = [];
a = u(ids_A).'*invA_u;

b = -2*(u(k) - u(ids_A).'*invA_B);

c = -(W(k,k) -  W(k, ids_A)*invA_B);

eta_roots = roots([a b c]); % A bound for the step size (eta)
if any(imag(eta_roots(:))) % If roots are imaginary then step size = 0
    step_size_bound = 0;
    if a>0
        error(['eta roots is complex and a > 0 --> i.e. inequiality' ...
            ' ax^2+bx+c<0 doesn''t hold!!'])
    end
elseif a == 0 && b ==0 % If a,b  == 0 then ineq always holds for c<0 and for all eta.
    if c<0
        step_size_bound = inf;
    else
        error('inequality: (%f)*eta^2 + (%f)*eta + (%f)\n < 0 is false',...
            a, b, c);
    end
    
elseif a ==0
    if b<0 && c<0
        step_size_bound = inf;
    elseif b>0 && c<0
        step_size_bound = -c/b;
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
    
    step_size_bound = max(eta_roots);
end

end