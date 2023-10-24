function layers = createSEBlock(inputSize, reductionRatio)
    % Tạo các layers cho SE Block
    % inputSize: kích thước đầu vào (số kênh)
    % reductionRatio: tỷ lệ giảm kích thước trong giai đoạn squeeze
    
    % Squeeze Stage
    layers = [
        globalAveragePooling2dLayer()
        fullyConnectedLayer(inputSize / reductionRatio)
        reluLayer()
        fullyConnectedLayer(inputSize)
        sigmoidLayer()
    ];
    
    % Excitation Stage
    excitationLayer = [
        multiplicationLayer(2)
    ];
    
    % Kết hợp các layers
    layers = [
        layers
        excitationLayer
    ];
end
