function filtered_printf(varargin)
% filtered_printf(flgShow, varargin)
% prints to the screen if flgShow == true

flgShow = varargin{1};
if flgShow == true
    builtin('fprintf', varargin{2:end});
end

end