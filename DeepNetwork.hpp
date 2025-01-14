#pragma once
#include"vector"
#include"Matrix.hpp"
#include"Layer.hpp"
class DeepNetwork
{

public:
    DeepNetwork(std::vector<int> topology, double learningRate);
    void setCurrentInput(std::vector<double> input);
    void printToConsole();
    void printWeightMatrices();
    void printBiases();
    void forwardPropogation();
    void backPropogation();
    Layer* GetLayer(int nth);

    void setErrors();
    void setTarget(std::vector<double> target);

    void printErrors();
    double getGlobalError();
    double lastEpoachError();
    void printHistErrors();
    void saveHistErrors();
    double getLearningRate();
    void setErrorDerivatives();
    std::vector<double> gethisterrors();
    void updateWeights();
    void gardientComputation();
    std::vector<GeneralMatrix::Matrix*> averageGradients(std::vector<std::vector<GeneralMatrix::Matrix*>>);
    void saveThisError(double error);


    std::vector<GeneralMatrix::Matrix*> GetGradientMatrices();

private:
    int topologySize;
    std::vector<int> topology;
    std::vector<Layer*> layers;
    std::vector<GeneralMatrix::Matrix*> weightMatrices;
    std::vector<GeneralMatrix::Matrix*> GradientMatrices;
    std::vector<double> input;
    std::vector<GeneralMatrix::Matrix*> BaisMatrices;
    double error;
    std::vector<double> target;
    std::vector<double> errors;
    std::vector<double> histErrors;
    std::vector<double> errorDerivatives;
    double learningRate;



};

