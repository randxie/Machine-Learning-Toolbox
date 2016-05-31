% Feedforward neural network designed for multi input - 1 hidden layer -
% single output system
classdef FF_Net
    properties %(GetAccess=private)
        WeightLayer1=[];
        WeightLayer2=[];
        BiasLayer1=[];
        BiasLayer2=[];
        TrainInput=[];
        TrainOutput=[];
        CurrentIn=0;
        CurrentOut=0;
        eta=0.2;
        NumOfInput=0;
        NumOfHiddenNode=0; %number of neuron
        NumOfOutput=0;
        TrainMean=0;
        TrainVar=0;
    end
    properties %Training parameters
        WB_Range=2;
        J_min=0.01;
        Momentum=0;
        NumOfTrainLoop=0; %number of training loop
        n_a=0.01;
        n_b=0.01;
        n_decay=500;
        TrainFcn='';
    end
    methods
        function obj=SetParameter(obj,DataIn,DataOut,Para)
            obj.TrainInput=DataIn;
            obj.TrainOutput=DataOut;
            obj.NumOfInput=size(DataIn,2);
            obj.NumOfOutput=size(DataOut,2);
            obj.NumOfHiddenNode=Para.NumOfHiddenNode;
            obj.TrainFcn=Para.TrainFcn;
            obj.NumOfTrainLoop=Para.NumOfTrainLoop;
            obj.TrainMean=Para.TrainMeanX;
            obj.TrainVar=Para.TrainVarX;
        end
        function obj=InitNet(obj)
            obj.CurrentIn=zeros(obj.NumOfInput,1);
            obj.CurrentOut=zeros(obj.NumOfOutput,1);
            obj.WeightLayer1=obj.WB_Range*(rand(obj.NumOfHiddenNode,obj.NumOfInput)-0.5);
            obj.BiasLayer1=obj.WB_Range*(rand(obj.NumOfHiddenNode,1)-0.5);
            obj.WeightLayer2=obj.WB_Range*(rand(1,obj.NumOfHiddenNode)-0.5);
            obj.BiasLayer2=obj.WB_Range*(rand(1,1)-0.5);
        end
        
        function [TotV,HiddenNodeV,OutputV]=Feedforward(obj)
            TotV=obj.WeightLayer1*obj.CurrentIn+obj.BiasLayer1;
            HiddenNodeV=obj.ActFun(TotV,obj.TrainFcn);
            OutputV=obj.WeightLayer2*HiddenNodeV+obj.BiasLayer2;
        end
        
        function [DeltaW1,DeltaB1,DeltaW2,DeltaB2]=CalculateDelta(obj)
            [TotV,HiddenV,SimOutput]=Feedforward(obj);
            DeltaB2=obj.eta*(obj.CurrentOut-SimOutput);
            DeltaW2=(DeltaB2*HiddenV)';
            DeltaB1=obj.eta*(obj.CurrentOut-SimOutput)*(obj.WeightLayer2'.*obj.ActFunPrime(TotV,HiddenV,obj.TrainFcn));
            DeltaW1=DeltaB1*obj.CurrentIn';
        end
        
        function obj=Update(obj)
            [DeltaW1,DeltaB1,DeltaW2,DeltaB2]=CalculateDelta(obj);
            obj.WeightLayer1=obj.WeightLayer1+DeltaW1;
            obj.BiasLayer1=obj.BiasLayer1+DeltaB1;
            obj.WeightLayer2=obj.WeightLayer2+DeltaW2;
            obj.BiasLayer2=obj.BiasLayer2+DeltaB2;
        end
        
        function [o]=ActFun(obj,tot,TrainFcn)
            if strcmp(TrainFcn,'logsig')==1
                o=1./(1+exp(-tot));
            elseif strcmp(TrainFcn,'tanh')==1
                o=tanh(tot);
            elseif strcmp(TrainFcn,'RBF')==1
                o=exp(-(tot-obj.TrainMean).^2/(2*obj.TrainVar^2));
            end          
        end
        
        function [res]=ActFunPrime(obj,tot,o,TrainFcn)
            if strcmp(TrainFcn,'logsig')==1
                res=o.*(1-o);
            elseif strcmp(TrainFcn,'tanh')==1
                res=1-o.^2;
            elseif strcmp(TrainFcn,'RBF')==1
                res=o.*(-(tot-obj.TrainMean)/obj.TrainVar^2);
            end
        end
        
        function obj=TrainLoop(obj)
            for j=1:obj.NumOfTrainLoop
                disp(['loop: ' num2str(j)]);
                obj.eta=obj.n_a+obj.n_b*exp(-j/obj.n_decay);
                for i=1:size(obj.TrainInput,1)
                    obj.CurrentIn=obj.TrainInput(i,:)';
                    obj.CurrentOut=obj.TrainOutput(i);
                    obj=obj.Update();  %Online update
                end
            end
        end
        
        function [SimTotOutput]=GenSimOutput(obj)
            SimTotOutput=zeros(size(obj.TrainInput,1),obj.NumOfOutput);
            for i=1:size(obj.TrainInput,1)
                obj.CurrentIn=obj.TrainInput(i,:);
                %obj.CurrentOut=obj.TrainOutput(i);
                obj.CurrentOut=obj.TrainOutput(i);
                [TotV,HiddenNodeV,OutputV]=Feedforward(obj);
                SimTotOutput(i,:)=OutputV';
            end
        end
    end 
end
