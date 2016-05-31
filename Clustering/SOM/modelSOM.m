classdef modelSOM
    properties
        weightMatrix=[];
        dataRegion={};
        storeSomName='';
    end
    methods
        function obj = modelSOM(sVec,somPara)
            % initialize network
            disp('initial SOM network');
            obj=obj.initialSOM(sVec,somPara);      
        end

        function obj=initialSOM(obj,sVec,somPara)
            switch somPara.initMethod
                case 'rand'
                    % assign random weight
                    obj.weightMatrix=(rand(somPara.xSize,somPara.ySize,somPara.numIn)-0.5);
                case 'kmean'
                    % assign weight by k mean (much faster)
                    [IDX, C] = kmeans(sVec, somPara.xSize*somPara.ySize);
                    disturbance=0.5*(rand(somPara.xSize,somPara.ySize,somPara.numIn)-0.5);
                    obj.weightMatrix=disturbance+reshape(C,[somPara.xSize,somPara.ySize,somPara.numIn]);
                otherwise
                    error ('modelSOM: no such initialization method');
            end
            obj.dataRegion=cell(somPara.xSize,somPara.ySize);
        end

        function obj=trainModel(obj,t,sVec,somPara)
            % online update model
            parfor i=1:size(sVec,1)
                dataIn_i=sVec(i,:)';
                BMU=findBMU(obj,dataIn_i,somPara);
                calDeltaWeight(obj,t,BMU,somPara);
            end
        end
        
        function obj=genRegion(obj,sVec,somPara)
            obj.dataRegion=cell(somPara.xSize,somPara.ySize);
            parfor i=1:size(sVec,1)
                dataIn_i=sVec(i,:)';
                BMU=findBMU(obj,dataIn_i,somPara);
                putBMUinRegion(obj,BMU);
            end
        end
        
        function []=putBMUinRegion(obj,BMU)
            tmp=obj.dataRegion{BMU.Ix,BMU.Iy};
            obj.dataRegion{BMU.Ix,BMU.Iy}=[tmp;BMU.dataIn_i'];
        end
        
        % search for BMU
        function [BMU]=findBMU(obj,dataIn_i,somPara)
            % calculate distance matrix
            tmp=(dataIn_i*ones(1,somPara.ySize))';
            tmp=reshape(tmp,[1,somPara.ySize,somPara.numIn]);
            distMtx=bsxfun(@minus,obj.weightMatrix,tmp);
            distMtx=sqrt(sum((distMtx.*distMtx),3));

            % find winner
            [Y,Ix]=min(distMtx);
            [Y,Iy]=min(Y);
            Ix=Ix(Iy);
            
            % output BMU
            BMU.Ix=Ix;  BMU.Iy=Iy;  
            BMU.dataIn_i=dataIn_i;

        end

        function []=calDeltaWeight(obj,t,BMU,somPara)
            tmp=(BMU.dataIn_i*ones(1,somPara.ySize))';
            tmp=reshape(tmp,[1,somPara.ySize,somPara.numIn]);
            deltaW=bsxfun(@minus,obj.weightMatrix,tmp);

            xPosMtx=repmat(1:1:somPara.xSize,[somPara.ySize,1]);
            yPosMtx=repmat((1:1:somPara.ySize)',[1,somPara.xSize]);
            
            distPosMtx=(xPosMtx-BMU.Ix).^2+(yPosMtx-BMU.Iy).^2;
            currR=obj.calNeibor(t,somPara);
            h=obj.calAlpha(t,somPara)*exp(-(distPosMtx)/(2*currR^2));
            
            h=repmat(h,[1,1,somPara.numIn]);
            updateOrNotMtx=repmat(distPosMtx<currR^2,[1 1 somPara.numIn]);
            dW=-h.*deltaW.*updateOrNotMtx;
            obj.weightMatrix=obj.weightMatrix+dW;
        end

        function [neiborhood]=calNeibor(obj,t,somPara)
            neiborhood=somPara.initR*exp(-t/somPara.tc);
        end

        function [alpha]=calAlpha(obj,t,somPara)
            alpha=somPara.alpha*exp(-t/somPara.tc);
        end
        
        function obj=PlotNetwork(obj,somPara)
            figure;
            for i=1:somPara.xSize
                for j=1:somPara.ySize
                    point=obj.weightMatrix(i,j,:);
                    plot(point(1),point(2),'ro');
                    hold on;
                end
            end
        end
    end
end