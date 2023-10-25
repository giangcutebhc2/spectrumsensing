numSubFrames = 40;        % corresponds to 40 ms
sampleRate = 61.44e6;     % Hz
frameDuration = numSubFrames*1e-3;    % seconds

%% Load trained-model
load('baseline.mat')
net_base = trainednetInfo{1,1};
load('improved_model.mat')
net_improved = trainednetInfo{1,1};

% Test
trainDir = fullfile(pwd,'TrainingData');
dataDir = fullfile(trainDir,'LTE_NR');
imds = imageDatastore(dataDir,'IncludeSubfolders',false,'FileExtensions','.png');

% Evaluate
classNames = ["NR" "LTE" "Noise"];
pixelLabelID = [127 255 0];

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

% Show
% 120 550 600 1000
pxdsResults_base = semanticseg(imds,net_base,"WriteLocation",tempdir);
pxdsResults_improved = semanticseg(imds,net_improved,"WriteLocation",tempdir);

pxdsTruth = pixelLabelDatastore(labelFiles,classNames,pixelLabelID);

imgIdx = 550;
trueLabels = readimage(pxdsTruth,imgIdx);
predicted_base = readimage(pxdsResults_base,imgIdx);
predicted_improved = readimage(pxdsResults_improved,imgIdx);

rcvdSpectrogram = readimage(imds,imgIdx);
figure
set(gcf, 'Position', [0, 100, 700, 1000]);  % Adjust the figure size as needed
helperCreateComparing(rcvdSpectrogram,trueLabels,predicted_base, predicted_improved, ...
   classNames,sampleRate,0,frameDuration)


