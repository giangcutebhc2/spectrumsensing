function helpcompare(signal,trueLabels,predictedLabels_1, predictedLabels_2,predictedLabels_3, predictedLabels_4, ...
  classNames,sr,fc,to)
%helperSpecSenseDisplayResults Spectrum sensing results
%   helperSpecSenseDisplayResults(P,TL,PL,C,FS,FC,TF) displays the receive
%   spectrogram, P, together with true labels, TL, and predicted labels,
%   TP. Possible class names are, C, sampling rate is FS, center frequency
%   is FC, and frame time is TF. 

%   Copyright 2021 The MathWorks, Inc.

numClasses = numel(classNames);
cmap = cool(numClasses);

if ~isempty(trueLabels)
  trueLabels = double(trueLabels);
end

predictedLabels_1 = double(predictedLabels_1);
predictedLabels_2 = double(predictedLabels_2);
predictedLabels_3 = double(predictedLabels_3);
predictedLabels_4 = double(predictedLabels_4);
t = linspace(-to,0,size(signal,1)) * 1e3;
f = (linspace(-sr/2,sr/2,size(signal,2)) + fc)/1e6;
freqDim = 2; % Put frequency on the x-axis

N = numel(classNames);
ticks = ((N-1)/3)*(2:N+1);

subplot(611)

% Flip the data. image functions will flip it again. Then by setting YDir
% to normal, we flip it back to the correct orientation.
signal = flipud(signal);
if freqDim == 2
  imagesc(f,t,signal)
else
  imagesc(t,f,signal)
end
set(gca,'YDir','normal')
a = colorbar;
colormap(a,parula(256))
% title('X', 'FontSize', 20, 'FontWeight', 'normal', 'Interpreter','latex');
if ~isempty(trueLabels)
  subplot(612)
  if freqDim == 2
    im = imagesc(f,t,trueLabels,[1 numClasses]);
  else
    im = imagesc(t,f,trueLabels,[1 numClasses]);
  end
  set(gca,'YDir','normal')
  im.Parent.Colormap = cmap;
  colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
    'TickLength',0,'TickLabelInterpreter','latex', 'FontSize', 18);
  % title('Ground truth', 'FontSize', 20, 'FontWeight', 'normal', 'Interpreter','latex');
end
if ~isempty(trueLabels)
  subplot(613)
else
  subplot(212)
end
predictedLabels_1 = flipud(predictedLabels_1);
if freqDim == 2
  im = imagesc(f,t,predictedLabels_1,[1 numClasses]);
else
  im = imagesc(t,f,predictedLabels_1,[1 numClasses]);
end
set(gca,'YDir','normal')
im.Parent.Colormap = cmap;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
  'TickLength',0,'TickLabelInterpreter','latex','FontSize', 18);
% title('Baseline', 'FontSize', 20, 'FontWeight', 'normal', 'Interpreter','latex');

if ~isempty(trueLabels)
  subplot(614)
else
  subplot(212)
end
predictedLabels_2 = flipud(predictedLabels_2);
if freqDim == 2
  im = imagesc(f,t,predictedLabels_2,[1 numClasses]);
else
  im = imagesc(t,f,predictedLabels_2,[1 numClasses]);
end
set(gca,'YDir','normal')
im.Parent.Colormap = cmap;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
  'TickLength',0,'TickLabelInterpreter','latex','FontSize', 18);
% title('Baseline + ATM', 'FontSize', 20, 'FontWeight', 'normal', 'Interpreter','latex');
if ~isempty(trueLabels)
  subplot(615)
else
  subplot(212)
end
predictedLabels_3 = flipud(predictedLabels_3);
if freqDim == 2
  im = imagesc(f,t,predictedLabels_3,[1 numClasses]);
else
  im = imagesc(t,f,predictedLabels_3,[1 numClasses]);
end
set(gca,'YDir','normal')
im.Parent.Colormap = cmap;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
  'TickLength',0,'TickLabelInterpreter','latex','FontSize', 18);
% title('Baseline + ADR', 'FontSize', 20, 'FontWeight', 'normal', 'Interpreter','latex');
if ~isempty(trueLabels)
  subplot(616)
else
  subplot(212)
end
predictedLabels_4 = flipud(predictedLabels_4);
if freqDim == 2
  im = imagesc(f,t,predictedLabels_4,[1 numClasses]);
else
  im = imagesc(t,f,predictedLabels_4,[1 numClasses]);
end
set(gca,'YDir','normal')
im.Parent.Colormap = cmap;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,...
  'TickLength',0,'TickLabelInterpreter','latex','FontSize', 18);
% title('Baseline + ATM + ADR', 'FontSize', 20, 'FontWeight', 'normal', 'Interpreter','latex');
end