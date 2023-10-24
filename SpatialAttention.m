classdef SpatialAttention < nnet.layer.Layer
    properties
        Conv
        Sigmoid
    end
    
    methods
        function layer = SpatialAttention(kernel_size)
            layer.Name = 'SpatialAttention';
            layer.Description = 'Spatial Attention Module';
            
            % Initialize convolutional layer
            layer.Conv = convolution2dLayer(kernel_size, 1, 'Padding', 'same');
            
            % Initialize sigmoid activation layer
            layer.Sigmoid = sigmoidLayer();
        end
        
        function Z = predict(layer, X)
            % Compute mean on spatial dimension
            avg_feat = mean(X, 3);
            
            % Compute max on spatial dimension
            max_feat = max(X, [], 3);
            
            % Concatenate average and max features along the channel dimension
            feat = cat(3, avg_feat, max_feat);
            
            % Apply convolution
            out_feat = predict(layer.Conv, feat);
            
            % Apply sigmoid activation
            attention = predict(layer.Sigmoid, out_feat);
            
            % Element-wise multiplication of attention and input
            Z = attention .* X;
        end
    end
end
