#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include"Neuron.h"
#include<vector>
#include"Matrix.hpp"
using namespace std;

class Layer
{
public:

    // Constructor for 
    Layer(int size);
    Layer(int size, bool last);
    void setVal(int index, double val);
    void setActivatedVal(int index, double val);

    GeneralMatrix::Matrix* convertTOMatrixVal();
    GeneralMatrix::Matrix* convertTOMatrixActivatedVal();
    GeneralMatrix::Matrix* convertTOMatrixDerivedVal();
    Layer* feedForward(GeneralMatrix::Matrix* LastWeights, GeneralMatrix::Matrix* LastBias, bool isFirst, bool isLast);
    vector<Neuron*> getNeurons();
    Neuron* getNeuron(int);
    int getSize();


private:
    int size;
    vector<Neuron*> neurons;
};


#endif