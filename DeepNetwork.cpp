#include "Layer.hpp"
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include"DeepNetwork.hpp"


DeepNetwork::DeepNetwork(vector<int> topology, double lr)
{
    this->learningRate = lr;
    this->topology = topology;
    this->topologySize = topology.size();

    for (int i = 0; i < topologySize; i++)
    {

        // If it is last layer treat it seperately
        if (i == topologySize - 1)
        {
            Layer* l = new Layer(topology[i], 1);
            this->layers.push_back(l);
        }

        // If it is not last layer
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
    // cout<<"This Iteration Error"<<endl;
    // for(auto err : this->errors)
    // {
    //     cout<<"Ex : "<<err<<"  ";
    // }
    // cout<<endl;

    // cout<<"Historical Errors"<<endl;
    // for(auto err : this->errors)
    // {
    //     cout<<"Eh : "<<err<<"  ";
    // }
    // cout<<endl;

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
    // Print the inputs to the DeepNetwork
    for (int i = 0; i < input.size(); i++)
    {
        cout << input.at(i) << "\t\t";
    }
    cout << endl;

    // Print the outputs to the DeepNetwork
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
    errorDerivatives.resize(errors.size());

    // -------------------------MSE---------------------------------------------
    //for (int i = 0; i < target.size(); i++)
    //{
    //    double req = target[i];
    //    double act = outputNeurons[i]->getActivatedVal();
    //    this->errors[i] = 0.5 * pow(abs((req - act)), 2);
    //    errorDerivatives[i] = act - req;
    //    this->error += errors[i];
    //}



    //------------------------Cross Entropy----------------------------------------
    for (int i = 0; i < target.size(); i++)
    {
        double act = target[i];
        double pred = outputNeurons[i]->getActivatedVal();

        double epsilon = 1e-15; // Small value to avoid instability
        pred = std::max(epsilon, std::min(1.0 - epsilon, pred));

        this->errors[i] = act * std::log(pred) + (1 - act) * std::log(1 - pred);

        // These are baseically the gadeints
        errorDerivatives[i] = -((act / pred) - ((1 - act) / (1 - pred)));

        // This is the total cross entropy error
        this->error += this->errors[i]; //
        // std::cout<<act<<"\t"<<pred<<"\t" << this->error << std::endl;
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

    GeneralMatrix::Matrix* gradients;
    GeneralMatrix::Matrix* DerivedValuesFromOtoH;
    GeneralMatrix::Matrix* lastGradient;
    GeneralMatrix::Matrix* tranposedWeightMatrices;
    GeneralMatrix::Matrix* hiddenDerived;
    int outputLayerIndex = this->topology.size() - 1;
    vector<GeneralMatrix::Matrix*> allGradients;

    /*
        -----------------------------------------VVIP NOTE ---------------------------------------------
        So turns out, using softmax with Cross Entropy is extermly good. When we used the sigmoid activation with cross entropy we used to calculate the gradiets as
        
        gradient_for_ith_otuput(gi) = Cross_entropy_loss_derivative(yi) x sigmoid_derivative(yi)

        but now as we are using softmax, the equation neatly cancels out to just gradient  = predicted - actual output
    */


    gradients = new GeneralMatrix::Matrix(
        1,
        this->topology.at(outputLayerIndex),
        false);
   
    
    
    size_t n = this->layers.at(outputLayerIndex)->getNeurons().size();
    vector<double> trueLabel = this->target;
    for (size_t i = 0; i < n; i++) {
        gradients->setVal(0, i, this->layers.at(outputLayerIndex)->getNeuron(i)->getActivatedVal() - trueLabel[i]);
    }

    GeneralMatrix::Matrix* temp = new GeneralMatrix::Matrix(1, this->topology.at(outputLayerIndex), false);
    allGradients.push_back(*temp + gradients);
    
    
    
    for (int i = (outputLayerIndex - 1); i > 0; i--)
    {
        GeneralMatrix::Matrix* t = new GeneralMatrix::Matrix(1, topology.at(i), false);
        lastGradient = new GeneralMatrix::Matrix(*gradients);
        delete gradients;

        tranposedWeightMatrices = this->weightMatrices.at(i)->tranpose();
        gradients = new GeneralMatrix::Matrix(
            1,
            this->topology.at(i),
            false);


        gradients = *lastGradient * tranposedWeightMatrices;

        hiddenDerived = this->layers.at(i)->convertTOMatrixDerivedVal();

        for (int colCounter = 0; colCounter < hiddenDerived->getNumCols(); colCounter++)
        {
            double g = gradients->getVal(0, colCounter) * hiddenDerived->getVal(0, colCounter);
            gradients->setVal(0, colCounter, g);
        }
        allGradients.push_back(*t + gradients);

        delete lastGradient;
        delete tranposedWeightMatrices;
        delete hiddenDerived;
    }

    this->GradientMatrices = allGradients;
}





void DeepNetwork::updateWeights()
{
    vector<GeneralMatrix::Matrix*> newWeights;
    vector<GeneralMatrix::Matrix*> newBiases;
    GeneralMatrix::Matrix* deltaWeights;
    GeneralMatrix::Matrix* gradients;

    GeneralMatrix::Matrix* gradientsTransposed;
    GeneralMatrix::Matrix* PreviousLayerActivatedVals;
    GeneralMatrix::Matrix* tempNewWeights;
    GeneralMatrix::Matrix* tempNewBiases;
    GeneralMatrix::Matrix* lastGradient;
    GeneralMatrix::Matrix* tranposedWeightMatrices;
    GeneralMatrix::Matrix* hiddenDerived;
    GeneralMatrix::Matrix* transposedHidden;
    int outputLayerIndex = topology.size() - 1;

    gradients = this->GradientMatrices[0];
    gradientsTransposed = gradients->tranpose();
    PreviousLayerActivatedVals = this->layers.at(outputLayerIndex - 1)->convertTOMatrixActivatedVal();
    deltaWeights = new GeneralMatrix::Matrix(
        gradientsTransposed->getNumRow(),
        PreviousLayerActivatedVals->getNumCols(),
        false);

    deltaWeights = *gradientsTransposed * PreviousLayerActivatedVals;

    tempNewWeights = new GeneralMatrix::Matrix(
        this->topology.at(outputLayerIndex - 1),
        this->topology.at(outputLayerIndex),
        false);


    for (int r = 0; r < this->topology.at(outputLayerIndex - 1); r++)
    {
        for (int c = 0; c < this->topology.at(outputLayerIndex); c++)
        {

            double originalValue = this->weightMatrices.at(outputLayerIndex - 1)->getVal(r, c);
            double deltaValue = deltaWeights->getVal(c, r);
            deltaValue = this->learningRate * deltaValue;

            tempNewWeights->setVal(r, c, (originalValue - deltaValue));
        }
    }

    // Update biases
    tempNewBiases = new GeneralMatrix::Matrix(
        1, // Bias is a row vector
        this->topology.at(outputLayerIndex),
        false);

    for (int c = 0; c < this->topology.at(outputLayerIndex); c++)
    {
        double originalBias = this->BaisMatrices.at(outputLayerIndex - 1)->getVal(0, c);
        double deltaBias = gradients->getVal(0, c); // Gradient for bias
        deltaBias = this->learningRate * deltaBias;

        tempNewBiases->setVal(0, c, (originalBias - deltaBias));
    }
    newBiases.push_back(new GeneralMatrix::Matrix(*tempNewBiases));


    newWeights.push_back(new GeneralMatrix::Matrix(*tempNewWeights));
    delete gradientsTransposed;
    delete PreviousLayerActivatedVals;
    delete tempNewWeights;
    delete tempNewBiases;
    delete deltaWeights;



    int gmctr = 1;

    for (int i = (outputLayerIndex - 1); i > 0; i--)
    {
        lastGradient = new GeneralMatrix::Matrix(*gradients);
        delete gradients;

        tranposedWeightMatrices = this->weightMatrices.at(i)->tranpose();

        gradients = new GeneralMatrix::Matrix(
            lastGradient->getNumRow(),
            tranposedWeightMatrices->getNumCols(),
            false);

        gradients = *lastGradient * tranposedWeightMatrices;

        hiddenDerived = this->layers.at(i)->convertTOMatrixDerivedVal();

        for (int colCounter = 0; colCounter < hiddenDerived->getNumCols(); colCounter++)
        {
            double g = gradients->getVal(0, colCounter) * hiddenDerived->getVal(0, colCounter);
            gradients->setVal(0, colCounter, g);
        }
        gradients = GradientMatrices[gmctr];
        gmctr++;

        if (i == 1)
        {
            PreviousLayerActivatedVals = this->layers.at(0)->convertTOMatrixVal();
        }
        else
        {
            PreviousLayerActivatedVals = this->layers.at(i - 1)->convertTOMatrixActivatedVal();
        }

        transposedHidden = PreviousLayerActivatedVals->tranpose();

        deltaWeights = new GeneralMatrix::Matrix(
            transposedHidden->getNumRow(),
            gradients->getNumCols(),
            false);

        deltaWeights = *transposedHidden * gradients;

        tempNewWeights = new GeneralMatrix::Matrix(
            this->weightMatrices.at(i - 1)->getNumRow(),
            this->weightMatrices.at(i - 1)->getNumCols(),
            false);

        // Update weights
        for (int r = 0; r < tempNewWeights->getNumRow(); r++)
        {
            for (int c = 0; c < tempNewWeights->getNumCols(); c++)
            {
                double originalValue = this->weightMatrices.at(i - 1)->getVal(r, c);
                double deltaValue = deltaWeights->getVal(r, c);

                deltaValue = this->learningRate * deltaValue;

                tempNewWeights->setVal(r, c, (originalValue - deltaValue));
            }
        }
        newWeights.push_back(new GeneralMatrix::Matrix(*tempNewWeights));

        // Update biases
        tempNewBiases = new GeneralMatrix::Matrix(
            1, // Bias is a row vector
            gradients->getNumCols(),
            false);

        for (int c = 0; c < gradients->getNumCols(); c++)
        {
            double originalBias = this->BaisMatrices.at(i - 1)->getVal(0, c);
            double deltaBias = gradients->getVal(0, c);
            deltaBias = this->learningRate * deltaBias;

            tempNewBiases->setVal(0, c, (originalBias - deltaBias));
        }
        newBiases.push_back(new GeneralMatrix::Matrix(*tempNewBiases));

        // Clean up
        delete lastGradient;
        delete tranposedWeightMatrices;
        delete hiddenDerived;
        delete PreviousLayerActivatedVals;
        delete transposedHidden;
        delete tempNewWeights;
        delete tempNewBiases;
        delete deltaWeights;
    }

    for (int i = 0; i < this->weightMatrices.size(); i++)
    {
        delete this->weightMatrices[i];
    }

    this->weightMatrices.clear();
    reverse(newWeights.begin(), newWeights.end());
    weightMatrices = newWeights;
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



std::vector<GeneralMatrix::Matrix*> DeepNetwork::GetGradientMatrices(){
    return this->GradientMatrices;
}