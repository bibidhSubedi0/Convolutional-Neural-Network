#pragma once
#include "all_includes.hpp"
#include <random>


namespace CNN_Matrix{
	class Matrix
	{
	public:
		static gridEntity convolute(gridEntity, gridEntity, int);
		static double genRandomNumber();
		static void randomize_all_values(gridEntity&,int,int);
		static gridEntity sum_of_all_matrix_elements(std::vector<gridEntity>);
		static double sum_of_all_elements(gridEntity);
		static void displayMatrix(gridEntity);
	};
}

namespace GeneralMatrix {
    class Matrix
    {
    public:

        Matrix(int numRows, int numCols, bool isRandom);
        double genRandomNumber();
        Matrix* tranpose();
        Matrix* operator *(Matrix*& A);
        Matrix* operator +(Matrix*& A);



        void setVal(int r, int c, double v);
        double getVal(int r, int c);

        void printToConsole();

        int getNumRow() { return this->numRows; }
        int getNumCols() { return this->numCols; }
        void divideByScalar(double scalar);

    private:
        int numRows;
        int numCols;
        std::vector < std:: vector<double >> values;
    };
}
