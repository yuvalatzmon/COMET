function [do_calc, varargout] = cond_load(filename, do_force, varargin)
%
% Load results from file if possible & allowed. 
%
% [do_calc, stats] = conditional_load(filename, do_force, varargin)
%
% Example: 
%   vars  = {'x','y'}
%   [do_calc,x,y] = cond_load(filename, do_force, vars{1:end});
%   if(do_calc < 1 ), return; end
%
% in other words: checks whether a file with analysis results for the 
% given variables already exists. if yes (and do_force is 0), then load the
% file and set a flag to skip the analysis (do_calc = 0) 

  
  if(do_force>0), 
    do_calc=1;
    n_vars = length(varargin);    
    for i_var = 1: n_vars
      var = varargin{i_var};
      cmd = sprintf('   varargout{%d}=[];',i_var,var);    
      eval(cmd);
    end

    return;
  end;
  
  
% Init var
  vars = varargin;
  n_vars = length(vars);
  for i_var = 1: n_vars 
    cmd = sprintf('%s = [];',vars{i_var}); eval(cmd);
  end
  varargout=[];

  if(nargout-1 ~= length(vars))
    fprintf('Warning: arguments mismatch in cond_load.m\n');
    fprintf('Calling stack is:\n');
    dbstack
    fprintf('End of stack\n');    
    error('arguments mismatch in cond_load.m');
  end
  
  % Check if file exists
  if(exist(filename,'file')>0), 
    do_calc=0;
  else
    do_calc=1;
  end;

  S.dummy=[];
  if(do_calc<1)
      try
	S = load(filename);
      catch
	fprintf('filename = "%s"\n',filename);
	fprintf('Warning: File exist, but could not be read\n');
	do_calc=1;
	keyboard	
      end
  end

  if(exist(filename,'file')==0)
    filename_prt = shorten_filename(filename);
    fprintf('File "%s" doesn''t exist. Calculate all vars.\n',filename_prt);
    for i_var = 1: n_vars
      var = vars{i_var};
      cmd = sprintf('   varargout{%d}=[];',i_var,var);    
      eval(cmd);
    end
    return
  end


  % File exists, Check if var was read
  for i_var = 1: n_vars
    var = vars{i_var};
    if(isfield(S,var)==0)
      do_calc=1;      
      fprintf('No variable "%s" found. Recalculate\n', var);
      cmd = sprintf('   varargout{%d}=[];',i_var);    
    else
      cmd = sprintf('   varargout{%d}=S.%s;',i_var,var);
    end 
    
    eval(cmd);
  end

return
end

% ===========================================
function filename_prt = shorten_filename(filename)  
% use this function to edit the way that files are printed
    filename_prt = filename;
end
  





