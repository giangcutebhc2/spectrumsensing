imageSize = [256 256];    % pixels
sampleRate = 61.44e6;     % Hz
numSubFrames = 40;        % corresponds to 40 ms
frameDuration = numSubFrames*1e-3;    % seconds
trainDir = fullfile(pwd,'TrainingData');

%numFramesPerStandard = 10000;

%helperSpecSenseTrainingData(numFramesPerStandard,imageSize,trainDir,numSubFrames,sampleRate);


%Load training data
imds = imageDatastore(trainDir,'IncludeSubfolders',false,'FileExtensions','.png');

classNames = ["NR" "LTE" "Noise"];
pixelLabelID = [127 255 0];
pxdsTruth = pixelLabelDatastore(trainDir,classNames,pixelLabelID,...
  'IncludeSubfolders',false,'FileExtensions','.hdf');

%Analyze Dataset Statistics
tbl = countEachLabel(pxdsTruth);
frequency = tbl.PixelCount/sum(tbl.PixelCount);
figure
bar(1:numel(classNames),frequency)
grid on
xticks(1:numel(classNames)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%Prepare Training, Validation, and Test Sets
[imdsTrain,pxdsTrain,imdsVal,pxdsVal] = helperSpecSensePartitionData(imds,pxdsTruth,[80 20]);
cdsTrain = combine(imdsTrain,pxdsTrain);
cdsVal = combine(imdsVal,pxdsVal);

% Apply a transform to resize the image and pixel label data to the desired
% size.
cdsTrain = transform(cdsTrain, @(data)preprocessTrainingData(data,imageSize));
cdsVal = transform(cdsVal, @(data)preprocessTrainingData(data,imageSize));

%Train Deep Neural Network
baseNetwork = 'resnet50';
lgraph = deeplabv3plusLayers(imageSize,numel(classNames),baseNetwork);

% %Điều chỉnh model nè
% 
% 
% 
% Tạo layers của SE Block
se256BlockLayers = createSEBlock(256, 16);
lgraph = addLayers(lgraph,se256BlockLayers);
lgraph = connectLayers(lgraph, 'bn2c_branch2c', 'gap');
lgraph = connectLayers(lgraph, 'bn2c_branch2c', 'layer_2/in2');
lgraph = disconnectLayers(lgraph, 'bn2c_branch2c', 'add_3/in1');
lgraph = connectLayers(lgraph, 'layer_2', 'add_3/in1');


%Balance Classes
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

% Hiển thị kiến trúc của model
analyzeNetwork(lgraph);
%analyzeNetwork(net);

%Select training options
opts = trainingOptions("sgdm",...
  MiniBatchSize = 20,...
  MaxEpochs = 20, ...
  LearnRateSchedule = "piecewise",...
  InitialLearnRate = 0.02,...
  LearnRateDropPeriod = 10,...
  LearnRateDropFactor = 0.1,...
  ValidationFrequency = 800,...
  ValidationData = cdsVal,...
  ValidationPatience = 10,...
  Shuffle="every-epoch",...
  OutputNetwork = "best-validation-loss",...
  Plots = 'training-progress');

[net,trainInfo] = trainNetwork(cdsTrain,lgraph_1,opts);


%Test
dataDir = fullfile(trainDir,'LTE_NR');
imds = imageDatastore(dataDir,'IncludeSubfolders',false,'FileExtensions','.png');
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir);

%Evaluate
pxdsTruth = pixelLabelDatastore(dataDir,classNames,pixelLabelID,...
  'IncludeSubfolders',false,'FileExtensions','.hdf');
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

%Save in file
trainednetInfo = {};
trainednetInfo{1,1} = net;
trainednetInfo{1,2} = metrics;
trainednetInfo{1,3} = trainInfo;
save('resnet50_dil.mat','trainednetInfo')

% measure performance at different SNR levels
%---------------------------------------------------------------
files = dir(fullfile(dataDir,'*.mat'));
dataFiles = {};
labelFiles = {};
for p=1:numel(files)
  load(fullfile(files(p).folder,files(p).name),'params');
  if params.SNRdB == 0
    [~,name] = fileparts(files(p).name);
    dataFiles = [dataFiles; fullfile(files(p).folder,[name '.png'])]; 
    labelFiles = [labelFiles; fullfile(files(p).folder,[name '.hdf'])]; 
  end
end
imds = imageDatastore(dataFiles);
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir, MiniBatchSize=5);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

trainednetInfo{2,1} = metrics;
save('resnet50_dil.mat','trainednetInfo')
%---------------------------------------------------------------
%---------------------------------------------------------------
files = dir(fullfile(dataDir,'*.mat'));
dataFiles = {};
labelFiles = {};
for p=1:numel(files)
  load(fullfile(files(p).folder,files(p).name),'params');
  if params.SNRdB == 20
    [~,name] = fileparts(files(p).name);
    dataFiles = [dataFiles; fullfile(files(p).folder,[name '.png'])]; 
    labelFiles = [labelFiles; fullfile(files(p).folder,[name '.hdf'])]; 
  end
end
imds = imageDatastore(dataFiles);
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir, MiniBatchSize=5);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

trainednetInfo{2,2} = metrics;
save('resnet50_dil.mat','trainednetInfo')
%---------------------------------------------------------------
%---------------------------------------------------------------
files = dir(fullfile(dataDir,'*.mat'));
dataFiles = {};
labelFiles = {};
for p=1:numel(files)
  load(fullfile(files(p).folder,files(p).name),'params');
  if params.SNRdB == 40
    [~,name] = fileparts(files(p).name);
    dataFiles = [dataFiles; fullfile(files(p).folder,[name '.png'])]; 
    labelFiles = [labelFiles; fullfile(files(p).folder,[name '.hdf'])]; 
  end
end
imds = imageDatastore(dataFiles);
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir, MiniBatchSize=5);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

trainednetInfo{2,3} = metrics;
save('resnet50_dil.mat','trainednetInfo')
%---------------------------------------------------------------
%---------------------------------------------------------------
files = dir(fullfile(dataDir,'*.mat'));
dataFiles = {};
labelFiles = {};
for p=1:numel(files)
  load(fullfile(files(p).folder,files(p).name),'params');
  if params.SNRdB == 60
    [~,name] = fileparts(files(p).name);
    dataFiles = [dataFiles; fullfile(files(p).folder,[name '.png'])]; 
    labelFiles = [labelFiles; fullfile(files(p).folder,[name '.hdf'])]; 
  end
end
imds = imageDatastore(dataFiles);
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir, MiniBatchSize=5);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

trainednetInfo{2,4} = metrics;
save('resnet50_dil.mat','trainednetInfo')
%---------------------------------------------------------------
%---------------------------------------------------------------
files = dir(fullfile(dataDir,'*.mat'));
dataFiles = {};
labelFiles = {};
for p=1:numel(files)
  load(fullfile(files(p).folder,files(p).name),'params');
  if params.SNRdB == 80
    [~,name] = fileparts(files(p).name);
    dataFiles = [dataFiles; fullfile(files(p).folder,[name '.png'])]; 
    labelFiles = [labelFiles; fullfile(files(p).folder,[name '.hdf'])]; 
  end
end
imds = imageDatastore(dataFiles);
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir, MiniBatchSize=5);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

trainednetInfo{2,5} = metrics;
save('resnet50_dil.mat','trainednetInfo')
%---------------------------------------------------------------
%---------------------------------------------------------------
files = dir(fullfile(dataDir,'*.mat'));
dataFiles = {};
labelFiles = {};
for p=1:numel(files)
  load(fullfile(files(p).folder,files(p).name),'params');
  if params.SNRdB == 100
    [~,name] = fileparts(files(p).name);
    dataFiles = [dataFiles; fullfile(files(p).folder,[name '.png'])]; 
    labelFiles = [labelFiles; fullfile(files(p).folder,[name '.hdf'])]; 
  end
end
imds = imageDatastore(dataFiles);
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir, MiniBatchSize=5);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

trainednetInfo{2,6} = metrics;
save('resnet50_dil.mat','trainednetInfo')
%---------------------------------------------------------------
%---------------------------------------------------------------
files = dir(fullfile(dataDir,'*.mat'));
dataFiles = {};
labelFiles = {};
for p=1:numel(files)
  load(fullfile(files(p).folder,files(p).name),'params');
  if params.SNRdB > 40
    [~,name] = fileparts(files(p).name);
    dataFiles = [dataFiles; fullfile(files(p).folder,[name '.png'])]; 
    labelFiles = [labelFiles; fullfile(files(p).folder,[name '.hdf'])]; 
  end
end
imds = imageDatastore(dataFiles);
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir, MiniBatchSize=5);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

trainednetInfo{3,1} = metrics;
save('resnet50_dil.mat','trainednetInfo')
%---------------------------------------------------------------


%Show
imgIdx = 1000;
rcvdSpectrogram = readimage(imds,imgIdx);
trueLabels = readimage(pxdsTruth,imgIdx);
predictedLabels = readimage(pxdsResults,imgIdx);
figure
helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predictedLabels, ...
  classNames,sampleRate,0,frameDuration)