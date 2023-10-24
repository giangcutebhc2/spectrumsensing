lgraph = layerGraph();

tempLayers = imageInputLayer([256 256 3],"Name","input_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_2")
    fullyConnectedLayer(16,"Name","fc_1")
    reluLayer("Name","relu")
    fullyConnectedLayer(256,"Name","fc_2")
    sigmoidLayer("Name","layer_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","layer_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","dec_c2","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn2")
    reluLayer("Name","dec_relu2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_15_relu")
    convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],128,"Name","res3c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_18_relu")
    convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_20_relu")
    convolution2dLayer([3 3],128,"Name","res3d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_26_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_27_relu")
    convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],256,"Name","res4c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_30_relu")
    convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([3 3],256,"Name","res4d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([3 3],256,"Name","res4e_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([3 3],256,"Name","res4f_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_39_relu")
    convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_41_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_44_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","aspp_Conv_1","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_1")
    reluLayer("Name","aspp_Relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_2","BiasLearnRateFactor",0,"DilationFactor",[6 6],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_2")
    reluLayer("Name","aspp_Relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_3","BiasLearnRateFactor",0,"DilationFactor",[12 12],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_3")
    reluLayer("Name","aspp_Relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_4","BiasLearnRateFactor",0,"DilationFactor",[18 18],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_4")
    reluLayer("Name","aspp_Relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","catAspp")
    convolution2dLayer([1 1],256,"Name","dec_c1","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn1")
    reluLayer("Name","dec_relu1")
    transposedConv2dLayer([8 8],256,"Name","dec_upsample1","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","dec_crop1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","dec_cat1")
    convolution2dLayer([3 3],256,"Name","dec_c3","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn3")
    reluLayer("Name","dec_relu3")
    convolution2dLayer([3 3],256,"Name","dec_c4","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn4")
    reluLayer("Name","dec_relu4")
    convolution2dLayer([1 1],3,"Name","scorer","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    transposedConv2dLayer([8 8],3,"Name","dec_upsample2","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    crop2dLayer("centercrop","Name","dec_crop2")
    softmaxLayer("Name","softmax-out")
    pixelClassificationLayer("Name","labels")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"input_1","conv1");
lgraph = connectLayers(lgraph,"input_1","dec_crop2/ref");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in2");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in1");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in1");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in2");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","gapool_2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","layer_2/in2");
lgraph = connectLayers(lgraph,"layer_1","layer_2/in1");
lgraph = connectLayers(lgraph,"layer_2","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"activation_10_relu","dec_c2");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in2");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in1");
lgraph = connectLayers(lgraph,"dec_relu2","dec_crop1/ref");
lgraph = connectLayers(lgraph,"dec_relu2","dec_cat1/in1");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in1");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in2");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in1");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in2");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in1");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in2");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in2");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in1");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in1");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in2");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in1");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in2");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in1");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in2");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in1");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in2");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in1");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in2");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in2");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in1");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in1");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in2");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in1");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in2");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_1");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_2");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_3");
lgraph = connectLayers(lgraph,"activation_49_relu","aspp_Conv_4");
lgraph = connectLayers(lgraph,"aspp_Relu_1","catAspp/in1");
lgraph = connectLayers(lgraph,"aspp_Relu_2","catAspp/in2");
lgraph = connectLayers(lgraph,"aspp_Relu_3","catAspp/in3");
lgraph = connectLayers(lgraph,"aspp_Relu_4","catAspp/in4");
lgraph = connectLayers(lgraph,"dec_upsample1","dec_crop1/in");
lgraph = connectLayers(lgraph,"dec_crop1","dec_cat1/in2");
lgraph = connectLayers(lgraph,"dec_upsample2","dec_crop2/in");