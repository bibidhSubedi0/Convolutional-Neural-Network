#include "../cnn/ImageInput.hpp"
#include "../cnn/Matrix.hpp"
#include "../cnn/ConvolutionLayers.hpp"
#include "../cnn/DeepNetwork.hpp"

int main()
{

    // =========================================================================
    // SECTION 1: Image Loading and Preprocessing
    // =========================================================================

    ImageInput img("resource/drawing.png", CV_8UC1);
    // img.showImage();  // Uncomment to visualize input image

    // Get matrixified pixel values for the images
    std::vector<std::vector<double>> pixelVals = img.getMatrixifiedPixelValues();

    // =========================================================================
    // SECTION 2: First Convolution Layer Setup and Feature Extraction
    // =========================================================================

    ConvolutionLayers firstConvLayer(pixelVals);

    // Apply all predefined filters to generate feature maps
    const auto& predefinedFilters = firstConvLayer.get_all_predefined_filter();
    for (size_t filterIdx = 0; filterIdx < predefinedFilters.size(); ++filterIdx)
    {
        gridEntity featureMap = firstConvLayer.apply_filter_universal(
            firstConvLayer.get_raw_input_image(),
            predefinedFilters[filterIdx],
            1  // Stride
        );
        firstConvLayer.get_feature_map().push_back(featureMap);
    }

    // Apply ReLU activation to all feature maps
    for (gridEntity& featureMap : firstConvLayer.get_feature_map())
    {
        firstConvLayer.activate_feature_map_using_RELU_universal(featureMap);
    }

    // Uncomment to visualize feature maps after activation
    // for (const gridEntity& featureMap : firstConvLayer.get_feature_map())
    // {
    //     ImageInput visualizer(featureMap);
    //     visualizer.showImage();
    // }

    // =========================================================================
    // SECTION 3: Max Pooling Layer
    // =========================================================================

    // Apply max pooling with 2x2 kernel to reduce spatial dimensions
    for (size_t i = 0; i < firstConvLayer.get_feature_map().size(); ++i)
    {
        gridEntity pooledMap = firstConvLayer.apply_pooling_univeral(
            firstConvLayer.get_feature_map()[i],
            2  // Pool size
        );
        firstConvLayer.get_pool_map().push_back(pooledMap);
    }

    // Uncomment to visualize pooled feature maps
    // for (const gridEntity& pooledMap : firstConvLayer.get_pool_map())
    // {
    //     ImageInput visualizer(pooledMap);
    //     visualizer.showImage();
    // }

    // =========================================================================
    // SECTION 4: Prepare Input Channels for Second Convolution Layer
    // =========================================================================

    // The vector<gridEntity> pool_maps IS THE INPUT CHANNEL for next layer
    firstConvLayer.get_input_channels() = firstConvLayer.get_pool_map();

    // =========================================================================
    // SECTION 5: Deep Neural Network Initialization
    // =========================================================================

    std::vector<double> targetOutput = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };  // For digit "5"

    const double neuralNetLearningRate = 0.01;

    // Define network topology: [input_size, hidden_size, output_size]
    std::vector<int> networkTopology = {
        363,                                       // Input layer size (flattened pooled features)
        400,                                       // Hidden layer size
        static_cast<int>(targetOutput.size())      // Output layer size
    };

    DeepNetwork neuralNetwork(networkTopology, neuralNetLearningRate);

    // =========================================================================
    // SECTION 6: Training Loop
    // =========================================================================

    const double convolutionLearningRate = 0.01;
    const int totalEpochs = 10;

    for (int epoch = 0; epoch < totalEpochs; ++epoch)
    {
        std::cout << "Training Epoch " << epoch << std::endl;

        // =====================================================================
        // STEP 6.1: Apply Second Convolution Layer (Trainable Filters)
        // =====================================================================

        // Apply the 'volumetricEntity training_filters' designated to the second convolution layer
        // and get the 'volumetricEntity output_features'
        std::vector<gridEntity> finalFilterMaps;
        const auto& trainingFilters = firstConvLayer.get_all_training_filter();

        for (size_t filterIdx = 0; filterIdx < trainingFilters.size(); ++filterIdx)
        {
            // Process each sheet of the training filter with corresponding input channel
            // First sheet of training filter to first sheet of input channel -> 1st filter map
            // Second sheet of training filter to second sheet of input channel -> 2nd filter map
            // ...
            // Nth sheet of training filter to nth sheet of input channel -> nth filter map

            std::vector<gridEntity> individualFilterMaps;
            const auto& inputChannels = firstConvLayer.get_input_channels();

            for (size_t channelIdx = 0; channelIdx < inputChannels.size(); ++channelIdx)
            {
                gridEntity filterMap = firstConvLayer.apply_filter_universal(
                    inputChannels[channelIdx],
                    trainingFilters[filterIdx][channelIdx],
                    1  // Stride
                );
                individualFilterMaps.push_back(filterMap);
            }

            // Sum all filter maps: 1st + 2nd + ... + nth = summed_filter_map
            gridEntity summedFilterMap = CNN_Matrix::Matrix::sum_of_all_matrix_elements(individualFilterMaps);
            finalFilterMaps.push_back(summedFilterMap);

            // Then go downward to apply activation and normalization
            firstConvLayer.activate_feature_map_using_RELU_universal(finalFilterMaps.back());

        }

        firstConvLayer.get_output_feature_maps() = finalFilterMaps;

        // =====================================================================
        // STEP 6.2: Apply Pooling to Output Features
        // =====================================================================

        firstConvLayer.get_final_pool_maps().clear();
        for (size_t i = 0; i < firstConvLayer.get_output_feature_maps().size(); ++i)
        {
            gridEntity pooledMap = firstConvLayer.apply_pooling_univeral(
                firstConvLayer.get_output_feature_maps()[i],
                2  // Pool size
            );
            firstConvLayer.get_final_pool_maps().push_back(pooledMap);
        }

        // =====================================================================
        // STEP 6.3: Flatten Pooled Layer for Neural Network Input
        // =====================================================================

        std::vector<double> flattenedVector;
        for (const auto& pooledMatrix : firstConvLayer.get_final_pool_maps())
        {
            for (const auto& row : pooledMatrix)
            {
                flattenedVector.insert(flattenedVector.end(), row.begin(), row.end());
            }
        }

        // =====================================================================
        // STEP 6.4: Forward Pass Through Deep Neural Network
        // =====================================================================

        // Extract dimensions from final pooled maps
        const int numChannels = firstConvLayer.get_final_pool_maps().size();  // Number of filters
        const int filterHeight = firstConvLayer.get_final_pool_maps()[0].size();  // Rows
        const int filterWidth = firstConvLayer.get_final_pool_maps()[0][0].size();  // Columns

        neuralNetwork.setCurrentInput(flattenedVector);
        neuralNetwork.setTarget(targetOutput);
        neuralNetwork.forwardPropogation();
        neuralNetwork.setErrors();
        neuralNetwork.gardientComputation();

        std::vector<GeneralMatrix::Matrix*> gradientMatrices = neuralNetwork.GetGradientMatrices();
        neuralNetwork.updateWeights();

        // =====================================================================
        // STEP 6.5: Backpropagate Gradients to Convolutional Layers
        // =====================================================================

        // Reshape the gradients in kernel form (i.e., for n channels, 
        // create arrays of width x height dimensions)
        GeneralMatrix::Matrix* lastGradient = gradientMatrices[gradientMatrices.size() - 1];
        GeneralMatrix::Matrix* transposedWeights = neuralNetwork.GetWeightMatrices()[0]->tranpose();
        GeneralMatrix::Matrix* requiredGradients = *lastGradient * transposedWeights;

        // Reshape flattened gradients back into 3D structure (channels x height x width)
        volumetricEntity pooledGradients;
        int currentColumn = 0;

        for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx)
        {
            gridEntity channelGradient;

            for (int row = 0; row < filterHeight; ++row)
            {
                std::vector<double> rowGradient;

                for (int col = 0; col < filterWidth; ++col)
                {
                    double gradientValue = requiredGradients->getVal(0, currentColumn);
                    rowGradient.push_back(gradientValue);
                    ++currentColumn;
                }

                channelGradient.push_back(rowGradient);
            }

            pooledGradients.push_back(channelGradient);
        }

        // =====================================================================
        // STEP 6.6: Unpool Gradients
        // =====================================================================

        // Unpool the gradients from their pooled map to get the gradients 
        // to the filter maps
        volumetricEntity unpooledGradients;

        for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx)
        {
            gridEntity unpooledMap = firstConvLayer.unpool_without_indices(
                pooledGradients[channelIdx],
                finalFilterMaps[channelIdx],
                2,  // Pool height
                2,  // Pool width
                2   // Stride
            );
            unpooledGradients.push_back(unpooledMap);
        }

        // =====================================================================
        // STEP 6.7: Compute Filter Gradients
        // =====================================================================

        // Get the gradients for filters from the gradients to the filter maps
        // we just obtained
        std::vector<gridEntity> inputChannelsForGradient = firstConvLayer.get_input_channels();
        std::vector<volumetricEntity> filterGradients = firstConvLayer.compute_filter_gradients(
            inputChannelsForGradient,
            unpooledGradients,
            1  // Stride
        );

        // =====================================================================
        // STEP 6.8: Update Convolutional Filters
        // =====================================================================

        // Update the filters based on the computed gradients
        firstConvLayer.update_filters_with_gradients(
            firstConvLayer.get_all_training_filter(),
            filterGradients,
            convolutionLearningRate
        );

        // =====================================================================
        // STEP 6.9: Display Training Progress
        // =====================================================================

        std::cout << "Error: " << neuralNetwork.getGlobalError() << std::endl;
        std::cout << "Softmax Values: " << std::endl;
        neuralNetwork.GetLayer(2)->convertTOMatrixActivatedVal()->printToConsole();
    }
        
    return 0;
}