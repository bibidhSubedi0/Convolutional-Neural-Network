#pragma once
#include<string>
#include "all_includes.hpp"
class ImageInput
{
	cv::Mat image;
	std::vector<std::vector<int>> pixel_values;
public:
	// Constructors
	ImageInput(std::string, int mode);
	ImageInput();

	// Set image if not done through constructor
	void setImage(std::string, int mode);

	// Print image to screen
	void showImage();

	// Change the image to matrix so i can write the convlution functions myself
	void matrixifyPixelValues();
	std::vector<std::vector<int>> getMatrixifiedPixelValues();

};