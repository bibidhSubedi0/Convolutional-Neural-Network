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
    std::vector<double> pixels; // Flattened 784 pixels
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

        // Remaining 784 values are pixel values
        while (std::getline(ss, value, ',')) {
            sample.pixels.push_back(std::stod(value) / 255.0); // Normalize to [0, 1]
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
// Helper Function: Create One-hot Encoded Target
// =========================================================================
std::vector<double> createOneHotTarget(int label) {
    std::vector<double> target(10, 0.0);
    target[label] = 1.0;
    return target;
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
    // SECTION 2: Initialize Deep Neural Network
    // =========================================================================

    const double learningRate = 0.1;

    std::vector<int> networkTopology = {
        784,           // Input layer (28x28 flattened pixels)
        128,           // Hidden layer 1
        64,            // Hidden layer 2
        10             // Output layer (10 digits)
    };

    DeepNetwork neuralNetwork(networkTopology, learningRate);

    std::cout << "Neural Network initialized with topology: ";
    for (size_t i = 0; i < networkTopology.size(); ++i) {
        std::cout << networkTopology[i];
        if (i < networkTopology.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;

    // =========================================================================
    // SECTION 3: Training Loop
    // =========================================================================

    const int totalEpochs = 20;

    for (int epoch = 0; epoch < totalEpochs; ++epoch) {
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
            // Forward Pass
            // ================================================================

            neuralNetwork.setCurrentInput(sample.pixels);
            neuralNetwork.setTarget(target);
            neuralNetwork.forwardPropogation();
            neuralNetwork.setErrors();

            double sampleError = neuralNetwork.getGlobalError();
            epochError += sampleError;

            // ================================================================
            // Get Prediction
            // ================================================================

            int predictedLabel = 0;
            double maxProb = 0.0;
            for (int j = 0; j < 10; ++j) {
                double prob = neuralNetwork.GetLayer(networkTopology.size() - 1)->getNeuron(j)->getActivatedVal();
                if (prob > maxProb) {
                    maxProb = prob;
                    predictedLabel = j;
                }
            }
            if (predictedLabel == sample.label) {
                correct++;
            }

            // ================================================================
            // Backward Pass
            // ================================================================

            neuralNetwork.gardientComputation();
            neuralNetwork.updateWeights();

            // ================================================================
            // Progress Display
            // ================================================================

            if ((i + 1) % 100 == 0) {
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
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}