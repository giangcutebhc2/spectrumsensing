% Đường dẫn tới thư mục để lưu ảnh đã chuyển đổi
outputDir = fullfile(pwd,'DenoiseData');

% Tạo thư mục nếu nó chưa tồn tại
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


% Duyệt qua tất cả các hình ảnh trong imds
for i = 1:length(imds.Files)
    img = readimage(imds, i); % Đọc hình ảnh thứ i từ imds
    [noisyR,noisyG,noisyB] = imsplit(img);
    dnnet = denoisingNetwork("dncnn");
    denoisedR = denoiseImage(noisyR,dnnet);
    denoisedG = denoiseImage(noisyG,dnnet);
    denoisedB = denoiseImage(noisyB,dnnet);
    denoisedRGB = cat(3,denoisedR,denoisedG,denoisedB);
    % Thêm mã lệnh xử lý khác nếu cần thiết
    % Lưu ảnh đã chuyển đổi vào thư mục mới
    % Đường dẫn mới sau khi thay đổi thư mục
    outputFileName = strrep(imds.Files{i}, 'TrainingData', 'DenoiseData');
    imwrite(denoisedRGB, outputFileName); % Lưu ảnh đã chuyển đổi
end
