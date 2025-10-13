#include "../cnn/Layer.hpp"
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "../cnn/DeepNetwork.hpp"

DeepNetwork::DeepNetwork(vector<int> topology, double lr)
{
    this->learningRate = lr;
    this->topology = topology;
    this->topologySize = topology.size();

    for (int i = 0; i < topologySize; i++)
    {
        if (i == topologySize - 1)
        {
            Layer* l = new Layer(topology[i], true);
            this->layers.push_back(l);
        }
        else {
            Layer* l = new Layer(topology[i]);
            this->layers.push_back(l);
        }
    }

    for (int i = 0; i < topologySize - 1; i++)
    {
        GeneralMatrix::Matrix* mw = new GeneralMatrix::Matrix(topology[i], topology[i + 1], true);
        this->weightMatrices.push_back(mw);

        GeneralMatrix::Matrix* mb = new GeneralMatrix::Matrix(1, topology[i + 1], true);
        this->BaisMatrices.push_back(mb);
    }

    histErrors.push_back(1);
}

double DeepNetwork::lastEpoachError()
{
    return histErrors[histErrors.size() - 1];
}

void DeepNetwork::printHistErrors()
{
    cout << "\n Printing Errors from all epochs:" << endl;
    for (int i = 0; i < this->histErrors.size(); i++)
    {
        cout << "Epoch: " << i << ", Error:" << histErrors.at(i) << " \n ";
    }
}

void DeepNetwork::saveHistErrors()
{
    ofstream outFile("error_vs_epoch.csv");
    if (outFile.is_open())
    {
        outFile << "Epoch,Error\n";
        for (size_t i = 0; i < this->histErrors.size(); ++i)
        {
            outFile << i << "," << histErrors[i] << "\n";
        }
        outFile.close();
    }
    else
    {
        cerr << "Unable to open file for writing.\n";
    }
}

vector<double> DeepNetwork::gethisterrors()
{
    return histErrors;
}

double DeepNetwork::getLearningRate()
{
    return learningRate;
}

void DeepNetwork::setTarget(vector<double> target)
{
    this->target = target;
}

void DeepNetwork::printErrors()
{
    cout << "Total Error : " << this->error << endl;
}

void DeepNetwork::setCurrentInput(vector<double> input)
{
    this->input = input;

    for (int i = 0; i < input.size(); i++)
    {
        this->layers[0]->setVal(i, input[i]);
        this->layers[0]->getNeuron(i)->Activate();
        this->layers[0]->getNeuron(i)->Derive();
    }
}

void DeepNetwork::printToConsole()
{
    for (int i = 0; i < input.size(); i++)
    {
        cout << input.at(i) << "\t\t";
    }
    cout << endl;

    for (int i = 0; i < layers.at(layers.size() - 1)->getSize(); i++)
    {
        cout << layers.at(layers.size() - 1)->getNeurons().at(i)->getActivatedVal() << "\t";
    }
}

Layer* DeepNetwork::GetLayer(int nth)
{
    return layers[nth];
}

void DeepNetwork::printWeightMatrices()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Weights for Hidden Layer : " << i + 1 << endl;
        weightMatrices[i]->printToConsole();
    }
}

void DeepNetwork::forwardPropogation()
{
    for (int i = 0; i < layers.size() - 1; i++)
    {
        layers[i + 1] = layers[i]->feedForward(weightMatrices[i], BaisMatrices[i], (i == 0), (i == layers.size() - 2));
    }
}

void DeepNetwork::printBiases()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Bias for Hidden Layer : " << i + 1 << endl;
        BaisMatrices[i]->printToConsole();
    }
}

void DeepNetwork::setErrors()
{
    if (this->target.size() == 0)
    {
        cerr << "No target found" << endl;
        assert(false);
    }

    if (target.size() != layers[layers.size() - 1]->getNeurons().size())
    {
        cerr << "The size of the target is not equal to the size of the output" << endl;
        assert(false);
    }

    errors.resize(target.size());
    this->error = 0;
    int outputLayerIndx = this->layers.size() - 1;
    vector<Neuron*> outputNeurons = this->layers[outputLayerIndx]->getNeurons();

    for (int i = 0; i < target.size(); i++)
    {
        double act = target[i];
        double pred = outputNeurons[i]->getActivatedVal();

        double epsilon = 1e-15;
        pred = std::max(epsilon, std::min(1.0 - epsilon, pred));

        this->errors[i] = -act * std::log(pred);
        this->error += this->errors[i];
    }
}

double DeepNetwork::getGlobalError()
{
    return this->error;
}

void DeepNetwork::saveThisError(double error)
{
    this->histErrors.push_back(error);
}

void DeepNetwork::gardientComputation()
{
    int outputLayerIndex = this->topology.size() - 1;

    // Clear old gradients
    for (auto g : this->GradientMatrices) {
        delete g;
    }
    this->GradientMatrices.clear();

    // Output layer gradient (softmax + cross-entropy derivative)
    GeneralMatrix::Matrix* gradients = new GeneralMatrix::Matrix(1, this->topology.at(outputLayerIndex), false);

    size_t n = this->layers.at(outputLayerIndex)->getNeurons().size();
    vector<double> trueLabel = this->target;
    for (size_t i = 0; i < n; i++) {
        double pred = this->layers.at(outputLayerIndex)->getNeuron(i)->getActivatedVal();
        gradients->setVal(0, i, pred - trueLabel[i]);
    }

    this->GradientMatrices.push_back(gradients);

    // Backpropagate through hidden layers
    for (int i = (outputLayerIndex - 1); i > 0; i--)
    {
        GeneralMatrix::Matrix* lastGradient = this->GradientMatrices.back();

        GeneralMatrix::Matrix* transposedWeightMatrices = this->weightMatrices.at(i)->tranpose();
        GeneralMatrix::Matrix* backpropGrad = *lastGradient * transposedWeightMatrices;

        GeneralMatrix::Matrix* hiddenDerived = this->layers.at(i)->convertTOMatrixDerivedVal();

        // Element-wise multiplication with derivatives
        GeneralMatrix::Matrix* newGradients = new GeneralMatrix::Matrix(1, hiddenDerived->getNumCols(), false);
        for (int colCounter = 0; colCounter < hiddenDerived->getNumCols(); colCounter++)
        {
            double g = backpropGrad->getVal(0, colCounter) * hiddenDerived->getVal(0, colCounter);
            newGradients->setVal(0, colCounter, g);
        }

        this->GradientMatrices.push_back(newGradients);

        delete transposedWeightMatrices;
        delete hiddenDerived;
        delete backpropGrad;
    }
}

std::vector<GeneralMatrix::Matrix*> DeepNetwork::GetWeightMatrices()
{
    return this->weightMatrices;
}

void DeepNetwork::updateWeights()
{
    vector<GeneralMatrix::Matrix*> newWeights;
    vector<GeneralMatrix::Matrix*> newBiases;

    // GradientMatrices[0] = output layer gradient
    // GradientMatrices[1] = last hidden layer gradient
    // ... and so on (reversed order)

    int numLayers = this->weightMatrices.size();

    for (int layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        // Get gradient for this layer (they're stored in reverse order)
        int gradIdx = layerIdx;
        GeneralMatrix::Matrix* gradients = this->GradientMatrices[gradIdx];

        // Get activations from previous layer
        int actualLayerIdx = numLayers - 1 - layerIdx;
        GeneralMatrix::Matrix* prevLayerActivations;

        if (actualLayerIdx == 0) {
            prevLayerActivations = this->layers.at(0)->convertTOMatrixVal();
        }
        else {
            prevLayerActivations = this->layers.at(actualLayerIdx)->convertTOMatrixActivatedVal();
        }

        // Compute weight gradients: prevActivations^T * gradient
        GeneralMatrix::Matrix* prevTransposed = prevLayerActivations->tranpose();
        GeneralMatrix::Matrix* deltaWeights = *prevTransposed * gradients;

        // Create new weight matrix
        GeneralMatrix::Matrix* tempNewWeights = new GeneralMatrix::Matrix(
            this->weightMatrices.at(actualLayerIdx)->getNumRow(),
            this->weightMatrices.at(actualLayerIdx)->getNumCols(),
            false);

        // Update weights with gradient clipping
        for (int r = 0; r < tempNewWeights->getNumRow(); r++)
        {
            for (int c = 0; c < tempNewWeights->getNumCols(); c++)
            {
                double originalValue = this->weightMatrices.at(actualLayerIdx)->getVal(r, c);
                double deltaValue = deltaWeights->getVal(r, c);

                // Gradient clipping
                deltaValue = std::max(-1.0, std::min(1.0, deltaValue));

                double update = this->learningRate * deltaValue;
                tempNewWeights->setVal(r, c, originalValue - update);
            }
        }
        newWeights.push_back(tempNewWeights);

        // Update biases
        GeneralMatrix::Matrix* tempNewBiases = new GeneralMatrix::Matrix(
            1,
            this->topology.at(actualLayerIdx + 1),
            false);

        for (int c = 0; c < tempNewBiases->getNumCols(); c++)
        {
            double originalBias = this->BaisMatrices.at(actualLayerIdx)->getVal(0, c);
            double deltaBias = gradients->getVal(0, c);

            // Gradient clipping
            deltaBias = std::max(-1.0, std::min(1.0, deltaBias));

            double update = this->learningRate * deltaBias;
            tempNewBiases->setVal(0, c, originalBias - update);
        }
        newBiases.push_back(tempNewBiases);

        delete prevTransposed;
        delete prevLayerActivations;
        delete deltaWeights;
    }

    // Replace old weights and biases
    for (int i = 0; i < this->weightMatrices.size(); i++)
    {
        delete this->weightMatrices[i];
        delete this->BaisMatrices[i];
    }

    this->weightMatrices.clear();
    this->BaisMatrices.clear();

    // Reverse because we computed them from output to input
    reverse(newWeights.begin(), newWeights.end());
    reverse(newBiases.begin(), newBiases.end());

    this->weightMatrices = newWeights;
    this->BaisMatrices = newBiases;
}

std::vector<GeneralMatrix::Matrix*> DeepNetwork::averageGradients(vector<vector<GeneralMatrix::Matrix*>> accgrad)
{
    size_t numColumns = accgrad[0].size();
    vector<GeneralMatrix::Matrix*> averageMatrices(numColumns, nullptr);

    for (size_t col = 0; col < numColumns; ++col)
    {
        averageMatrices[col] = new GeneralMatrix::Matrix(*accgrad[0][col]);

        for (int i = 0; i < averageMatrices[col]->getNumRow(); ++i)
        {
            for (int j = 0; j < averageMatrices[col]->getNumCols(); ++j)
            {
                averageMatrices[col]->setVal(i, j, 0.0);
            }
        }
    }

    for (int avgmahunuparne = 0; avgmahunuparne < numColumns; avgmahunuparne++)
    {
        for (int i = 0; i < accgrad.size(); i++)
        {
            averageMatrices[avgmahunuparne] = *averageMatrices[avgmahunuparne] + accgrad[i][avgmahunuparne];
        }
    }

    size_t numInputs = accgrad.size();
    for (size_t col = 0; col < numColumns; ++col)
    {
        averageMatrices[col]->divideByScalar(numInputs);
    }
    this->GradientMatrices.resize(averageMatrices.size());
    this->GradientMatrices = averageMatrices;
    return averageMatrices;
}

std::vector<GeneralMatrix::Matrix*> DeepNetwork::GetGradientMatrices() {
    return this->GradientMatrices;
}