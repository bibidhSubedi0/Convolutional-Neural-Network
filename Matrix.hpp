#pragma once
#include "all_includes.hpp"
#include <random>



class Matrix
{
public:
	static gridEntity convolute(gridEntity, gridEntity, int);
	static double genRandomNumber();
	static void randomize_all_values(gridEntity&,int,int);
	static gridEntity sum_of_all_matrix_elements(std::vector<gridEntity>);
	static double sum_of_all_elements(gridEntity);
};

