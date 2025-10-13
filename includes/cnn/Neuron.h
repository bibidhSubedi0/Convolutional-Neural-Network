#pragma once
#include<iostream>
#include<vector>
using namespace std;

class Neuron
{
public:

    Neuron(double val);
    void setVal(double v);
    // Activation Function

    //Fast Sigmoide Function f(x) = x/(1+|x|)
    // Derivative => f'(x) = f(x) *(1-f(x))
    void Activate();
    void Derive();
    void ActivateFinal();
    void DeriveFinal();
    void ActivateSoftmax(std::vector<Neuron*>);
    void setDerivedVal(double val);

    void setActivatedVal(double val); // LOLLL FUCK MY LIFE -> ISTG I NEED A SYSTEM DESGIN COURSE

    // Getter
    double getVal();
    double getActivatedVal();
    double getDerivedVal();

private:
    double val;
    double activatedVal; // After passing through sigmoide
    double derivedVal; // approx derivative of activacted val

};

