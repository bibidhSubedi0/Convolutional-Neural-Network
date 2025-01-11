#include "Layer.hpp"

Layer::Layer(int size)
{
    this->size = size;
    for (int i = 0; i < size; i++)
    {
        Neuron* n = new Neuron(0.00);
        n->Activate();
        n->Derive();
        this->neurons.push_back(n);
    }
}

Layer::Layer(int size, bool last)
{
    this->size = size;
    for (int i = 0; i < size; i++)
    {
        Neuron* n = new Neuron(0.00);
        n->ActivateFinal();
        n->DeriveFinal();
        this->neurons.push_back(n);
    }
}

int Layer::getSize()
{
    return size;
}

GeneralMatrix::Matrix* Layer::convertTOMatrixVal()
{
    GeneralMatrix::Matrix* m = new GeneralMatrix::Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < neurons.size(); i++)
    {
        m->setVal(0, i, this->neurons[i]->getVal());
    }
    return m;
}
GeneralMatrix::Matrix* Layer::convertTOMatrixActivatedVal()
{
    GeneralMatrix::Matrix* m = new GeneralMatrix::Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < neurons.size(); i++)
    {
        m->setVal(0, i, this->neurons[i]->getActivatedVal());
    }
    return m;
}
GeneralMatrix::Matrix* Layer::convertTOMatrixDerivedVal()
{
    GeneralMatrix::Matrix* m = new GeneralMatrix::Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < neurons.size(); i++)
    {
        m->setVal(0, i, this->neurons[i]->getDerivedVal());
    }
    return m;
}

Layer* Layer::feedForward(GeneralMatrix::Matrix* Weights, GeneralMatrix::Matrix* bias, bool isFirst, bool isLast)
{


    GeneralMatrix::Matrix* this_layer_val;

    if (isFirst)
    {
        this_layer_val = convertTOMatrixVal();
    }
    else {
        this_layer_val = convertTOMatrixActivatedVal();
    }

    GeneralMatrix::Matrix* TransFormedMat = new GeneralMatrix::Matrix(1, Weights->getNumCols(), false);
    GeneralMatrix::Matrix* z = *this_layer_val * Weights;
    GeneralMatrix::Matrix* zWithBias = *z + bias;

    // Put the Calculated Values in the Layer in form of vector rather then matrix
    if (!isLast) {

        Layer* temp = new Layer(Weights->getNumCols());

        for (int i = 0; i < Weights->getNumCols(); i++)
        {
            temp->setVal(i, zWithBias->getVal(0, i));
            temp->getNeuron(i)->Activate();
            temp->getNeuron(i)->Derive();
        }

        return temp;
    }
    else // Last ho vane
    {
        Layer* temp = new Layer(Weights->getNumCols(), 1);

        for (int i = 0; i < Weights->getNumCols(); i++)
        {
            temp->setVal(i, zWithBias->getVal(0, i));
            temp->getNeuron(i)->ActivateFinal();
            temp->getNeuron(i)->DeriveFinal();
        }
        return temp;
    }

}

Neuron* Layer::getNeuron(int pos)
{


    return neurons.at(pos);
};

vector<Neuron*> Layer::getNeurons()
{
    return neurons;
}

void Layer::setVal(int i, double v)
{
    this->neurons[i]->setVal(v);
}