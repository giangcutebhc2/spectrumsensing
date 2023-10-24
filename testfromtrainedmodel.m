%% Load trained-model
net_base = trainednetInfo{1,1};
net_att = trainednetInfo{1,1};
net_dil = trainednetInfo{1,1};
net_ful = trainednetInfo{1,1};

mob_base = trainednetInfo{1,1};
mob_att = trainednetInfo{1,1};
mob_dil = trainednetInfo{1,1};
mob_full = trainednetInfo{1,1};

r18_base = trainednetInfo{1,1};
r18_att = trainednetInfo{1,1};
r18_dil = trainednetInfo{1,1};
r18_full = trainednetInfo{1,1};

% Test
trainDir = fullfile(pwd,'TrainingData');
dataDir = fullfile(trainDir,'LTE_NR');
imds = imageDatastore(dataDir,'IncludeSubfolders',false,'FileExtensions','.png');
pxdsResults_base = semanticseg(imds,net_base,"WriteLocation",tempdir);
pxdsResults_att = semanticseg(imds,net_att,"WriteLocation",tempdir);
pxdsResults_dil = semanticseg(imds,net_dil,"WriteLocation",tempdir);
pxdsResults_ful = semanticseg(imds,net_ful,"WriteLocation",tempdir);

% Evaluate
classNames = ["NR" "LTE" "Noise"];
pixelLabelID = [127 255 0];
% Evaluate
% pxdsTruth = pixelLabelDatastore(dataDir,classNames,pixelLabelID,...
%   'IncludeSubfolders',false,'FileExtensions','.hdf');
% metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);
% 
% %% Confusion matrix
% cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
%   classNames, Normalization='row-normalized');
% cm.Title = 'Normalized Confusion Matrix';

% Measure performance at different SNR levels
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
% pxdsResults = semanticseg(imds,net_ful,"WriteLocation",tempdir, MiniBatchSize=20);
% 
% pxdsResults_base = semanticseg(imds,net_base,"WriteLocation",tempdir);
% pxdsResults_att = semanticseg(imds,net_att,"WriteLocation",tempdir);
% pxdsResults_dil = semanticseg(imds,net_dil,"WriteLocation",tempdir);
% pxdsResults_ful = semanticseg(imds,net_ful,"WriteLocation",tempdir);
% pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
% imgIdx = 175;
% rcvdSpectrogram = readimage(imds,imgIdx);
% trueLabels = readimage(pxdsTruth,imgIdx);
% predictedLabels = readimage(pxdsResults,imgIdx);
% figure
% set(gcf, 'Position', [0, 100, 700, 500]);  % Adjust the figure size as needed
% helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predictedLabels, ...
%   classNames,sampleRate,0,frameDuration)

% metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);


% % Show
% imgIdx = 2;
numSubFrames = 40;        % corresponds to 40 ms
sampleRate = 61.44e6;     % Hz
frameDuration = numSubFrames*1e-3;    % seconds
% rcvdSpectrogram = readimage(imds,imgIdx);
% trueLabels = readimage(pxdsTruth,imgIdx);
% predicted_base = readimage(pxdsResults_base,imgIdx);
% predicted_att = readimage(pxdsResults_att,imgIdx);
% predicted_dil = readimage(pxdsResults_dil,imgIdx);
% predicted_ful = readimage(pxdsResults_ful,imgIdx);
% figure
% helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predictedLabels, ...
%   classNames,sampleRate,0,frameDuration)
% 
% helpcompare(rcvdSpectrogram,trueLabels,predicted_base,predicted_att,predicted_dil, predicted_ful, ...
%    classNames,sampleRate,0,frameDuration)
% % %  
pxdsResults_base_mob = semanticseg(imds,mob_base,"WriteLocation",tempdir);
pxdsResults_att_mob = semanticseg(imds,mob_att,"WriteLocation",tempdir);
pxdsResults_dil_mob = semanticseg(imds,mob_dil,"WriteLocation",tempdir);
pxdsResults_ful_mob = semanticseg(imds,mob_full,"WriteLocation",tempdir);
predicted_base = readimage(pxdsResults_base_mob,imgIdx);
predicted_att = readimage(pxdsResults_att_mob,imgIdx);
predicted_dil = readimage(pxdsResults_dil_mob,imgIdx);
predicted_ful = readimage(pxdsResults_ful_mob,imgIdx);
figure
helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predictedLabels, ...
  classNames,sampleRate,0,frameDuration)

helpcompare(rcvdSpectrogram,trueLabels,predicted_base,predicted_att,predicted_dil, predicted_ful, ...
   classNames,sampleRate,0,frameDuration)

% 
pxdsResults_base = semanticseg(imds,r18_base,"WriteLocation",tempdir);
pxdsResults_att = semanticseg(imds,r18_att,"WriteLocation",tempdir);
pxdsResults_dil = semanticseg(imds,r18_dil,"WriteLocation",tempdir);
pxdsResults_ful = semanticseg(imds,r18_full,"WriteLocation",tempdir);
predicted_base = readimage(pxdsResults_base,imgIdx);
predicted_att = readimage(pxdsResults_att,imgIdx);
predicted_dil = readimage(pxdsResults_dil,imgIdx);
predicted_ful = readimage(pxdsResults_ful,imgIdx);
figure
helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predictedLabels, ...
  classNames,sampleRate,0,frameDuration)

helpcompare(rcvdSpectrogram,trueLabels,predicted_base,predicted_att,predicted_dil, predicted_ful, ...
   classNames,sampleRate,0,frameDuration)

% Show
% 120 550 600 1000
imgIdx = 550;
rcvdSpectrogram = readimage(imds,imgIdx);
trueLabels = readimage(pxdsTruth,imgIdx);
% pxdsResults_base = semanticseg(imds,net_base,"WriteLocation",tempdir);
% pxdsResults_att = semanticseg(imds,net_att,"WriteLocation",tempdir);
% pxdsResults_dil = semanticseg(imds,net_dil,"WriteLocation",tempdir);
% pxdsResults_ful = semanticseg(imds,net_ful,"WriteLocation",tempdir);
pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);
trueLabels = readimage(pxdsTruth,imgIdx);
predicted_base = readimage(pxdsResults_base,imgIdx);
predicted_att = readimage(pxdsResults_att,imgIdx);
predicted_dil = readimage(pxdsResults_dil,imgIdx);
predicted_ful = readimage(pxdsResults_ful,imgIdx);
figure
set(gcf, 'Position', [0, 100, 700, 1000]);  % Adjust the figure size as needed
helpcompare(rcvdSpectrogram,trueLabels,predicted_base,predicted_att,predicted_dil, predicted_ful, ...
   classNames,sampleRate,0,frameDuration)
% 
% helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predicted_base, ...
%    classNames,sampleRate,0,frameDuration)
% 
% pxdsResults_base_mob = semanticseg(imds,mob_base,"WriteLocation",tempdir);
% pxdsResults_att_mob = semanticseg(imds,mob_att,"WriteLocation",tempdir);
% pxdsResults_dil_mob = semanticseg(imds,mob_dil,"WriteLocation",tempdir);
% pxdsResults_ful_mob = semanticseg(imds,mob_full,"WriteLocation",tempdir);
predicted_base = readimage(pxdsResults_base_mob,imgIdx);
predicted_att = readimage(pxdsResults_att_mob,imgIdx);
predicted_dil = readimage(pxdsResults_dil_mob,imgIdx);
predicted_ful = readimage(pxdsResults_ful_mob,imgIdx);
figure
set(gcf, 'Position', [850, 100, 700, 1000]);  % Adjust the figure size as needed
helpcompare(rcvdSpectrogram,trueLabels,predicted_base,predicted_att,predicted_dil, predicted_ful, ...
   classNames,sampleRate,0,frameDuration)
% 
% pxdsResults_base_r18 = semanticseg(imds,r18_base,"WriteLocation",tempdir);
% pxdsResults_att_r18 = semanticseg(imds,r18_att,"WriteLocation",tempdir);
% pxdsResults_dil_r18 = semanticseg(imds,r18_dil,"WriteLocation",tempdir);
% pxdsResults_ful_r18 = semanticseg(imds,r18_full,"WriteLocation",tempdir);
predicted_base = readimage(pxdsResults_base_r18,imgIdx);
predicted_att = readimage(pxdsResults_att_r18,imgIdx);
predicted_dil = readimage(pxdsResults_dil_r18,imgIdx);
predicted_ful = readimage(pxdsResults_ful_r18,imgIdx);
figure
set(gcf, 'Position', [1700, 100, 700, 1000]);  % Adjust the figure size as needed
% helperSpecSenseDisplayResults(rcvdSpectrogram,trueLabels,predictedLabels, ...
%   classNames,sampleRate,0,frameDuration)
helpcompare(rcvdSpectrogram,trueLabels,predicted_base,predicted_att,predicted_dil, predicted_ful, ...
   classNames,sampleRate,0,frameDuration)

result_compare{1,1} = imds;
save('result.mat','result_compare')
