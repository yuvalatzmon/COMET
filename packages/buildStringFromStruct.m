
function resultString = buildStringFromStruct(object, seperatorChar)
    resultString = handleObject(object, seperatorChar,'');
end


function resultString = handleObject(object, seperatorChar, parentName)
    if ischar(object)
       resultString = sprintf('%s_%s',parentName,object);
    elseif isnumeric(object)
       resultString = sprintf('%s_%g',parentName,object);
    elseif islogical(object)
       resultString = sprintf('%s_%d',parentName,object);
    elseif isstruct(object)
        names = fieldnames(object);
        if strcmp(parentName,'')
            newName = names{1};
        else
            newName = [parentName,'.',names{1}];
        end
        fieldValue = handleObject(object.(names{1}), seperatorChar, newName );
        resultString = sprintf('%s' ,  fieldValue );
        
        for i=2:length(names)
            if strcmp(parentName,'')
                newName = names{i};
            else
                newName = [parentName,'.',names{i}];
            end
            fieldValue = handleObject(object.(names{i}), seperatorChar, newName );
            resultString = sprintf('%s%s%s' , resultString, seperatorChar,  fieldValue );
        end
    end
end