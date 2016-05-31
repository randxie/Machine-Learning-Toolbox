function [OutMtx]=ReadPattern(FileName)
delimiterIn = ',';
headerlinesIn = 1;
Mtx = importdata(FileName,delimiterIn,headerlinesIn);
OutMtx=Mtx.data(:,1:2);
end