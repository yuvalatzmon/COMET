function bool_res = is_pattern_in_str(pattern, str)
 
if isempty(strfind(str, pattern))
    bool_res = 0;
else
    bool_res = 1;
end

    