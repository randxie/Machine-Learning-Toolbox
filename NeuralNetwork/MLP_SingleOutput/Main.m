%Test multi input - single output condition
clc;
clear;
tic;
FileName='pattern.tsn';
OutMtx=ReadPattern(FileName);
DataIn=OutMtx(:,1:end-1);
DataOut=OutMtx(:,end);

%For logsig, 50000 epoch, 11 neuron, 0.01+0.02 works
%For tanh, 80000 epoch, 11 neuron, 0.01+0.02 works
%For RBF, 90000 epoch, 11 neuron, 0.01+0.02 works
Para.NumOfHiddenNode=11;
Para.TrainFcn='RBF';
Para.NumOfTrainLoop=90000;
Para.TrainMeanX=mean(DataIn);
Para.TrainVarX=var(DataIn);
           
Network=FF_Net;
Network=Network.SetParameter(DataIn,DataOut,Para);
Network=Network.InitNet();
Network=Network.TrainLoop();
[res]=Network.GenSimOutput();
plot(DataIn,DataOut,'b',DataIn,res,'r');
toc;


