#include "ImageInput.hpp"

int main()
{
	
	ImageInput* img = new ImageInput("drawing.png", CV_8UC1);
	img->showImage();
	std::vector<std::vector<int>> pixelVals = img->getMatrixifiedPixelValues();

	// Go through the pixels;
	for (int i = 0; i < pixelVals.size(); i++)
	{
		for (int j = 0; j < pixelVals.size(); j++)
		{
			std::cout << pixelVals.at(i).at(j) << " ";
		}

		std::cout << "\n\n\n\n";
	}
}