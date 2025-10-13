#include "../cnn/ImageInput.hpp"
#include "../cnn/Matrix.hpp"
#include "../cnn/ConvolutionLayers.hpp"
#include "../cnn/DeepNetwork.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

// =========================================================================
// Helper Function: Load MNIST CSV Data
// =========================================================================
struct MNISTSample {
    int label;
    std::vector<std::vector<double>> pixels; // 28x28 image
};

std::vector<MNISTSample> loadMNISTFromCSV(const std::string& filepath, int maxSamples = -1) {
    std::vector<MNISTSample> dataset;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return dataset;
    }

    std::string line;
    // Skip header line if present
    std::getline(file, line);

    int sampleCount = 0;
    while (std::getline(file, line) && (maxSamples == -1 || sampleCount < maxSamples)) {
        std::stringstream ss(line);
        std::string value;

        MNISTSample sample;

        // First value is the label
        std::getline(ss, value, ',');
        sample.label = std::stoi(value);

        // Remaining 784 values are pixel values (28x28)
        std::vector<double> flatPixels;
        while (std::getline(ss, value, ',')) {
            flatPixels.push_back(std::stod(value) / 255.0); // Normalize to [0, 1]
        }

        // Convert flat vector to 28x28 matrix
        sample.pixels.resize(28);
        for (int i = 0; i < 28; ++i) {
            sample.pixels[i].resize(28);
            for (int j = 0; j < 28; ++j) {
                sample.pixels[i][j] = flatPixels[i * 28 + j];
            }
        }

        dataset.push_back(sample);
        sampleCount++;

        if (sampleCount % 1000 == 0) {
            std::cout << "Loaded " << sampleCount << " samples..." << std::endl;
        }
    }

    file.close();
    std::cout << "Total samples loaded: " << dataset.size() << std::endl;
    return dataset;
}

// =========================================================================
// Helper Function: Create Onehot Encoded Target
// =========================================================================
std::vector<double> createOneHotTarget(int label) {
    std::vector<double> target(10, 0.0);
    target[label] = 1.0;
    return target;
}

// =========================================================================
// Helper Function: Process Single Image Through Conv Layers
// =========================================================================
std::vector<double> processImageThroughConvLayers(
    ConvolutionLayers& convLayer,
    const std::vector<std::vector<double>>& pixelVals
) 
{
    // Clear previous data
    convLayer.get_feature_map().clear();
    convLayer.get_pool_map().clear();
    convLayer.get_output_feature_maps().clear();
    convLayer.get_final_pool_maps().clear();

    // Update raw image
    convLayer.get_raw_input_image() = pixelVals;

    // Apply predefined filters
    const auto& predefinedFilters = convLayer.get_all_predefined_filter();
    for (size_t filterIdx = 0; filterIdx < predefinedFilters.size(); ++filterIdx) {
        gridEntity featureMap = convLayer.apply_filter_universal(
            convLayer.get_raw_input_image(),
            predefinedFilters[filterIdx],
            1
        );
        convLayer.get_feature_map().push_back(featureMap);
    }

    // Apply ReLU activation
    for (gridEntity& featureMap : convLayer.get_feature_map()) {
        convLayer.activate_feature_map_using_RELU_universal(featureMap);
    }

    // Apply max pooling
    for (size_t i = 0; i < convLayer.get_feature_map().size(); ++i) {
        gridEntity pooledMap = convLayer.apply_pooling_univeral(
            convLayer.get_feature_map()[i],
            2
        );
        convLayer.get_pool_map().push_back(pooledMap);
    }

    // Set input channels for second conv layer
    convLayer.get_input_channels() = convLayer.get_pool_map();

    // Apply trainable filters (second conv layer)
    std::vector<gridEntity> finalFilterMaps;
    const auto& trainingFilters = convLayer.get_all_training_filter();

    for (size_t filterIdx = 0; filterIdx < trainingFilters.size(); ++filterIdx) {
        std::vector<gridEntity> individualFilterMaps;
        const auto& inputChannels = convLayer.get_input_channels();

        for (size_t channelIdx = 0; channelIdx < inputChannels.size(); ++channelIdx) {
            gridEntity filterMap = convLayer.apply_filter_universal(
                inputChannels[channelIdx],
                trainingFilters[filterIdx][channelIdx],
                1
            );
            individualFilterMaps.push_back(filterMap);
        }

        gridEntity summedFilterMap = CNN_Matrix::Matrix::sum_of_all_matrix_elements(individualFilterMaps);
        finalFilterMaps.push_back(summedFilterMap);
        convLayer.activate_feature_map_using_RELU_universal(finalFilterMaps.back());
    }

    convLayer.get_output_feature_maps() = finalFilterMaps;

    // Apply pooling to output features
    for (size_t i = 0; i < convLayer.get_output_feature_maps().size(); ++i) {
        gridEntity pooledMap = convLayer.apply_pooling_univeral(
            convLayer.get_output_feature_maps()[i],
            2
        );
        convLayer.get_final_pool_maps().push_back(pooledMap);
    }

    // Flatten for neural network input
    std::vector<double> flattenedVector;
    for (const auto& pooledMatrix : convLayer.get_final_pool_maps()) {
        for (const auto& row : pooledMatrix) {
            flattenedVector.insert(flattenedVector.end(), row.begin(), row.end());
        }
    }

    return flattenedVector;
}

int main()
{
    // =========================================================================
    // SECTION 1: Load MNIST Dataset
    // =========================================================================

    std::cout << "Loading MNIST training data..." << std::endl;
    std::vector<MNISTSample> trainingData = loadMNISTFromCSV("resource/mnist_train.csv", 500);

    if (trainingData.empty()) {
        std::cerr << "Failed to load training data. Exiting." << std::endl;
        return -1;
    }

    // =========================================================================
    // SECTION 2: Initialize Convolutional Layers
    // =========================================================================

    // Create dummy 28x28 image for initialization
    std::vector<std::vector<double>> dummyImage(28, std::vector<double>(28, 0.0));
    ConvolutionLayers convLayer(dummyImage);

    // =========================================================================
    // SECTION 3: Calculate Network Input Size
    // =========================================================================

    // Process one sample to determine flattened size
    std::vector<double> sampleFlattened = processImageThroughConvLayers(convLayer, trainingData[0].pixels);
    int inputSize = sampleFlattened.size();

    std::cout << "Calculated input size: " << inputSize << std::endl;

    // =========================================================================
    // SECTION 4: Initialize Deep Neural Network
    // =========================================================================

    const double neuralNetLearningRate = 0.1;   // Increased for faster learning
    const double convolutionLearningRate = 0.0001; // Increased for faster learning

    std::vector<int> networkTopology = {
        inputSize,     // Input layer (flattened conv features)
        256,           // Hidden layer (increased from 128)
        10             // Output layer (10 digits)
    };

    double currentNNLearningRate = neuralNetLearningRate;
    double currentConvLearningRate = convolutionLearningRate;

    DeepNetwork neuralNetwork(networkTopology, neuralNetLearningRate);

    // =========================================================================
    // SECTION 5: Training Loop
    // =========================================================================

    const int totalEpochs = 10;
    const int batchSize = 16;


    for (int epoch = 0; epoch < totalEpochs; ++epoch) {


        // In your epoch loop, after each epoch:
        if ((epoch + 1) % 3 == 0) {
            currentNNLearningRate *= 0.9;     // Decay by 10%
            currentConvLearningRate *= 0.9;
            std::cout << "Learning rate decayed to: NN=" << currentNNLearningRate
                << ", Conv=" << currentConvLearningRate << std::endl;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Training Epoch " << epoch + 1 << "/" << totalEpochs << std::endl;
        std::cout << "========================================" << std::endl;

        double epochError = 0.0;
        int correct = 0;

        for (size_t i = 0; i < trainingData.size(); ++i) {
            // Get current sample
            MNISTSample& sample = trainingData[i];
            std::vector<double> target = createOneHotTarget(sample.label);

            // ================================================================
            // Forward Pass Through Conv Layers
            // ================================================================

            std::vector<double> flattenedVector = processImageThroughConvLayers(convLayer, sample.pixels);

            // ================================================================
            // Forward Pass Through Neural Network
            // ================================================================

            neuralNetwork.setCurrentInput(flattenedVector);
            neuralNetwork.setTarget(target);
            neuralNetwork.forwardPropogation();
            neuralNetwork.setErrors();

            double sampleError = neuralNetwork.getGlobalError();
            epochError += sampleError;

            // Check if prediction is correct
            int predictedLabel = 0;
            double maxProb = 0.0;
            for (int j = 0; j < 10; ++j) {
                double prob = neuralNetwork.GetLayer(2)->getNeuron(j)->getActivatedVal();
                if (prob > maxProb) {
                    maxProb = prob;
                    predictedLabel = j;
                }
            }
            if (predictedLabel == sample.label) {
                correct++;
            }

            // FIXED SECTION: Replace the backpropagation section in your main loop

            // ================================================================
            // Backward Pass (FIXED VERSION)
            // ================================================================

            neuralNetwork.gardientComputation();
            std::vector<GeneralMatrix::Matrix*> gradientMatrices = neuralNetwork.GetGradientMatrices();
            neuralNetwork.updateWeights();

            // Backpropagate to conv layers
            const int numChannels = convLayer.get_final_pool_maps().size();
            const int filterHeight = convLayer.get_final_pool_maps()[0].size();
            const int filterWidth = convLayer.get_final_pool_maps()[0][0].size();

            GeneralMatrix::Matrix* lastGradient = gradientMatrices[gradientMatrices.size() - 1];
            GeneralMatrix::Matrix* transposedWeights = neuralNetwork.GetWeightMatrices()[0]->tranpose();
            GeneralMatrix::Matrix* requiredGradients = *lastGradient * transposedWeights;

            // Reshape gradients back to 3D structure
            volumetricEntity pooledGradients;
            int currentColumn = 0;

            for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
                gridEntity channelGradient;
                for (int row = 0; row < filterHeight; ++row) {
                    std::vector<double> rowGradient;
                    for (int col = 0; col < filterWidth; ++col) {
                        double gradientValue = requiredGradients->getVal(0, currentColumn);
                        rowGradient.push_back(gradientValue);
                        ++currentColumn;
                    }
                    channelGradient.push_back(rowGradient);
                }
                pooledGradients.push_back(channelGradient);
            }

            // Unpool gradients
            volumetricEntity unpooledGradients;
            for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
                gridEntity unpooledMap = convLayer.unpool_without_indices(
                    pooledGradients[channelIdx],
                    convLayer.get_output_feature_maps()[channelIdx],
                    2, 2, 2
                );
                unpooledGradients.push_back(unpooledMap);
            }

            // CRITICAL FIX: Apply ReLU derivative before computing filter gradients
            // Store pre-activation values if needed, or apply derivative here
            for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
                convLayer.apply_relu_derivative(
                    unpooledGradients[channelIdx],
                    convLayer.get_output_feature_maps()[channelIdx]
                );
            }

            // Compute filter gradients with cross-correlation
            std::vector<gridEntity> inputChannelsForGradient = convLayer.get_input_channels();
            std::vector<volumetricEntity> filterGradients = convLayer.compute_filter_gradients(
                inputChannelsForGradient,
                unpooledGradients,
                1
            );

            if (i == 0 && epoch == 0) {  // Only for first sample of first epoch
                double maxGrad = 0.0, minGrad = 0.0, sumGrad = 0.0;
                int count = 0;

                for (const auto& filterGrad : filterGradients) {
                    for (const auto& channelGrad : filterGrad) {
                        for (const auto& row : channelGrad) {
                            for (double val : row) {
                                maxGrad = std::max(maxGrad, val);
                                minGrad = std::min(minGrad, val);
                                sumGrad += val;
                                count++;
                            }
                        }
                    }
                }

                std::cout << "Gradient stats - Min: " << minGrad
                    << ", Max: " << maxGrad
                    << ", Avg: " << (sumGrad / count) << std::endl;
            }

            // Update convolutional filters with clipping
            convLayer.update_filters_with_gradients(
                convLayer.get_all_training_filter(),
                filterGradients,
                convolutionLearningRate
            );

            delete transposedWeights;
            delete requiredGradients;

            // ================================================================
            // Progress Display
            // ================================================================

            if ((i + 1) % 50 == 0) {  // Report every 50 samples instead of 100
                double avgError = epochError / (i + 1);
                double accuracy = (100.0 * correct) / (i + 1);
                std::cout << "  Sample " << (i + 1) << "/" << trainingData.size()
                    << " | Avg Error: " << avgError
                    << " | Accuracy: " << accuracy << "%" << std::endl;
            }
        }

        // ====================================================================
        // Epoch Summary
        // ====================================================================

        double avgEpochError = epochError / trainingData.size();
        double epochAccuracy = (100.0 * correct) / trainingData.size();

        std::cout << "\n--- Epoch " << (epoch + 1) << " Summary ---" << std::endl;
        std::cout << "Average Error: " << avgEpochError << std::endl;
        std::cout << "Accuracy: " << epochAccuracy << "%" << std::endl;
        std::cout << "Correct: " << correct << "/" << trainingData.size() << std::endl;

        if ((epoch + 1) % 3 == 0) {
            currentNNLearningRate *= 0.9;     // Decay by 10%
            currentConvLearningRate *= 0.9;
            std::cout << "Learning rate decayed to: NN=" << currentNNLearningRate
                << ", Conv=" << currentConvLearningRate << std::endl;
        }

    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}