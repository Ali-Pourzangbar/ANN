clear all
clc
close all
%% %% Data Loading
%
%     input = {'BOD','COD','TSS'};
    OutputName = input('please enter the name of the output (BOD,COD,TSS) = ','s');    
%     if OutputNum ==1
        if strcmpi(OutputName,'BOD')
           Output_num=1;
           Data = xlsread('biowin_withMalfunction.xlsx',1);
%            Data = xlsread('Sensor_Lab_datasets_4Validation.xlsx',1);
        elseif strcmpi(OutputName,'COD')
           Output_num=2;
           Data = xlsread('biowin_withMalfunction.xlsx',3);
        elseif strcmpi(OutputName,'TSS')
           Output_num=3;
           Data = xlsread('biowin_withMalfunction.xlsx',5); % sheet 5 for 6 params; sheet 6 for 11 parameters
        end
        index = Output_num -3;
        X = Data(1:end,1:end-3);
        T = Data(1:end,end+index:end+index);   %% BOD, COD, TSS (end-2:end)
        NeuNet = ANN_AP(X,T,OutputName);
%     else
%         for i=1:OutputNum
%             OutputName{1,i} = input{1,i};
%         end
%         X = Data(1:end-3,1:end);
%         T = Data(end-2:end,1:end);   %% BOD, COD, TSS (end-2:end)
%         ANN_AP(X,T,OutputName)
%     end

%% Creating the ANN
function net = ANN_AP(X,T,OutputName)
% Data number; input data number; output data number
    DataNum   = size(X,1);
    InputNum  = size(X,2);
    OutputNum = size(T,2);
% 	net = fitnet([6 8 5]); % Hidden layer1=9 neurons; Hidden layer2= 8 neurons; Hidden layer3= 5 neurons; Output layer=1 neuron (based on the number of outputs)
    if strcmpi(OutputName,'BOD')
           net = feedforwardnet ([12 9 5]); % HL1=4 NEURONS; HL2=6 NEURONS; HL3=4 NEURONS; OUTPUT LAYER=outputs
    elseif strcmpi(OutputName,'COD')
        net = feedforwardnet ([9 8 6]); 
    elseif strcmpi(OutputName,'TSS')
        net = feedforwardnet ([9 8 6]);
    end
    net.trainParam.epochs=1000;
    net.trainParam.lr=0.05;
    net.trainParam.goal=0.005;
    net.trainParam.show=50;
    net.trainParam.max_fail=500;
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',1)); % to get constant result
%   net.performFcn='mse';  %Name of a network performance function %type help nnperformance
%   net.layers{1}.transferFcn = 'poslin';
% 	net.layers{2}.transferFcn = 'poslin';
    
% close(f)
% Set up Division of Data for Training, Validation, Testing
% 	net.divideFcn = 'divideblock'; % Divide targets into three sets using blocks of indices
    net.divideParam.trainRatio = 60/100; % try also j values for selecting train data points
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 25/100;
    net.trainFcn = 'trainlm';

    X = transpose(X);
    T = transpose(T);
    
    [net,tr] = train(net,X,T);
    if strcmpi(OutputName,'BOD')
       genFunction(net,'ANN_FcnBOD');
    elseif strcmpi(OutputName,'COD')
       genFunction(net,'ANN_FcnCOD');
    elseif strcmpi(OutputName,'TSS')
       genFunction(net,'ANN_FcnTSS');
    end
    nntraintool
% nntraintool('close')
% figure
% plotperform(tr)

    trainX = X(:,tr.trainInd);   % measured train input parameters
    trainT = T(:,tr.trainInd);   % measured train output parameters

    valX = X(:,tr.valInd);   % measured validation input parameters
    valT = T(:,tr.valInd);   % measured validation output parameters

    testX = X(:,tr.testInd);    % measured test input parameters
    testT = T(:,tr.testInd);   % measured test output parameters

%% Assesment
%Trainging data points
trainY = net(trainX);   % predicted by the developed ANN network
valY   = net(valX);
testY  = net(testX);

for i=1:OutputNum
    % Correlation Coefficient 
    CCTrain(1,i) = corr(trainY(i,:)',trainT(i,:)'); % two vectors must be vertical
    CCTest(1,i)  = corr(testY(i,:)',testT(i,:)');    
    CCFull(1,i)  = corr([trainY(i,:),valY(i,:),testY(i,:)]',[trainT(i,:),valT(i,:),testT(i,:)]');
    
    % Root Mean Square Error 
    RmseTrain(1,i) = sqrt(mse(trainY(i,:)',trainT(i,:)'));
    RmseTest(1,i)  = sqrt(mse(testY(i,:)',testT(i,:)'));
    RmseFull(1,i)  = sqrt(mse([trainY(i,:),valY(i,:),testY(i,:)]',[trainT(i,:),valT(i,:),testT(i,:)]'));
    
    % Scatter Index (SI) 
    SITrain(1,i) = sqrt(mse(trainY(i,:)',trainT(i,:)'))/mean(trainT(i,:)');
    SITest(1,i)  = sqrt(mse(testY(i,:)',testT(i,:)'))/mean(testT(i,:)');
    SIFull(1,i)  = sqrt(mse([trainY(i,:),valY(i,:),testY(i,:)]',[trainT(i,:),valT(i,:),testT(i,:)]'))/mean([trainT(i,:),valT(i,:),testT(i,:)]');
    
    % BIAS 
    BiasTrain(1,i) = sum(trainY(i,:)'- trainT(i,:)');
    BiasTest(1,i)  = sum(testY(i,:)'- testT(i,:)');
    BiasFull(1,i)  = sum([trainY(i,:),valY(i,:),testY(i,:)]'-[trainT(i,:),valT(i,:),testT(i,:)]');   
end
%% Writting the statistical parameters in a table
    Data_type = {'Train Dataset';'Test Dataset';'Full Dataset'};
for i=1:OutputNum
    CC (:,i)   = [CCTrain(1,i)  ;CCTest(1,i)  ;CCFull(1,i)  ];
    RMSE(:,i)  = [RmseTrain(1,i);RmseTest(1,i);RmseFull(1,i)];
    SI(:,i)    = [SITrain(1,i)  ;SITest(1,i)  ;SIFull(1,i)  ];
    BIAS(:,i)  = [BiasTrain(1,i);BiasTest(1,i);BiasFull(1,i)];
    Statistical_Indices = table(Data_type, CC, RMSE, SI, BIAS)
end
    filename = 'Statistical_Indices.xlsx';
    writetable(Statistical_Indices,filename,'Sheet',1,'Range','A1')
    
%% Display the results
% Ploting measured and predicted data points against time

for i=1:OutputNum
    figure
    if OutputNum==1
       dis = 4;
    elseif OutputNum==2
        if i==1
           dis = 8;
        elseif i==2
           dis = 4;           
        end
    elseif OutputNum==3
        if i==1
           dis = 5;
        elseif i==2
           dis = 8;
        elseif i==3
           dis = 4;
        end
    end
        plot([trainT(i,:),valT(i,:),testT(i,:)],'r','linewidth',2)
        hold on
        scatter([1:DataNum],[trainT(i,:),valT(i,:),testT(i,:)],30,'g','filled','linewidth',2)
        hold on
        plot([trainY(i,:),valY(i,:),testY(i,:)],'--k','linewidth',2)
        legend(['Measured', OutputName],['Measured (represented by points) ',OutputName],['Predicted',OutputName])
        xlabel('Data Number')
        ylabel(['Measured and predicted',OutputName])
        title(['Measured V.S. predicted (by the developed ANN model) ',OutputName,'  for Full dataset'])
        grid on
        grid minor
        box on
        legend('Location','northwest')
        xticks(0:500:DataNum)
        yticks(floor(min([trainT(i,:),valT(i,:),testT(i,:)])):dis:ceil(max([trainT(i,:),valT(i,:),testT(i,:)])))
end
        
        
   % SCATTER PLOTS FOR TRAIN, TEST, AND FULL DATASET
for i=1:OutputNum
    figure
    if i==1
        dis=4;
    elseif i==2
        dis=6;
    elseif i==3
        dis=2;
    end
    subplot(1,3,1)   %train data points
        t = floor(min([trainT(i,:),valT(i,:),testT(i,:)])):ceil(max([trainT(i,:),valT(i,:),testT(i,:)]));
        plot(t,t,'r','linewidth',2)
        hold on
        plot(trainT(i,:),trainY(i,:),'ok','MarkerSize',8,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
        axis equal
        axis([floor(min([trainT(i,:),valT(i,:),testT(i,:)])),ceil(max([trainT(i,:),valT(i,:),testT(i,:)])),floor(min([trainT(i,:),valT(i,:),testT(i,:)])),ceil(max([trainT(i,:),valT(i,:),testT(i,:)]))])
        xlabel(['Measured ' ,OutputName])
        ylabel(['Predicted ',OutputName])
        ax.FontSize = 12;
        grid on
        grid minor
        box on
        ALI= [OutputName,' Training data'];
        legend('Fit Line', ALI)
        legend('Location','northwest')
        xticks(floor(min([trainT(i,:),valT(i,:),testT(i,:)])):dis:ceil(max([trainT(i,:),valT(i,:),testT(i,:)])))
        yticks(floor(min([trainT(i,:),valT(i,:),testT(i,:)])):dis:ceil(max([trainT(i,:),valT(i,:),testT(i,:)])))
        
    subplot(1,3,2)
        t = floor(min([trainT(i,:),valT(i,:),testT(i,:)])):ceil(max([trainT(i,:),valT(i,:),testT(i,:)]));
        plot(t,t,'r','linewidth',2)
        hold on
        plot(testT(i,:),testY(i,:),'ok','MarkerSize',8,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
        axis equal
        axis([floor(min([trainT(i,:),valT(i,:),testT(i,:)])),ceil(max([trainT(i,:),valT(i,:),testT(i,:)])),floor(min([trainT(i,:),valT(i,:),testT(i,:)])),ceil(max([trainT(i,:),valT(i,:),testT(i,:)]))])
        xlabel(['Measured ' ,OutputName])
        ylabel(['Predicted ',OutputName])
        ax.FontSize = 12;
        grid on
        grid minor
        box on
        legend('Fit Line',[OutputName,'  Testing data'])
        legend('Location','northwest')
        xticks(floor(min([trainT(i,:),valT(i,:),testT(i,:)])):dis:ceil(max([trainT(i,:),valT(i,:),testT(i,:)])))
        yticks(floor(min([trainT(i,:),valT(i,:),testT(i,:)])):dis:ceil(max([trainT(i,:),valT(i,:),testT(i,:)])))
        
        
    subplot(1,3,3)
        t = floor(min([trainT(i,:),valT(i,:),testT(i,:)])):ceil(max([trainT(i,:),valT(i,:),testT(i,:)]));
        plot(t,t,'r','linewidth',2)
        hold on
        plot([trainT(i,:),valT(i,:),testT(i,:)],[trainY(i,:),valT(i,:),testY(i,:)],'ok','MarkerSize',8,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
        axis equal
        axis([floor(min([trainT(i,:),valT(i,:),testT(i,:)])),ceil(max([trainT(i,:),valT(i,:),testT(i,:)])),floor(min([trainT(i,:),valT(i,:),testT(i,:)])),ceil(max([trainT(i,:),valT(i,:),testT(i,:)]))])
        xlabel(['Measured ' ,OutputName])
        ylabel(['Predicted ',OutputName])
        ax.FontSize = 12;
        grid on
        grid minor
        box on
        legend('Fit Line',[OutputName,'  Full data'])
        legend('Location','northwest')
        xticks(floor(min([trainT(i,:),valT(i,:),testT(i,:)])):dis:ceil(max([trainT(i,:),valT(i,:),testT(i,:)])))
        yticks(floor(min([trainT(i,:),valT(i,:),testT(i,:)])):dis:ceil(max([trainT(i,:),valT(i,:),testT(i,:)])))
end

% figure
% for i=1:OutputNum
%     plotperform (tr)
% end
        
        
for i=1:OutputNum
    figure
    ploterrhist(trainT(i,:) - trainY(i,:),'train',testT(i,:) - testY(i,:),'test','bins',30);
    title(['Error = Target - Predicted for ',OutputName])
    set(gca,'FontSize',14,'FontWeight','Bold')
    grid minor
    box on;
end
        
    f = view(net);
end