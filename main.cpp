#include "ImageInput.hpp"
#include "ConvolutionLayers.hpp"

int main()
{
	
	ImageInput* img = new ImageInput("drawing.jpg", CV_8UC1);
	img->showImage();
	std::vector<std::vector<double>> pixelVals = img->getMatrixifiedPixelValues();



	// Start the first convolution Layer
	ConvolutionLayers l1(pixelVals);
	for (int i = 0; i < 5; i++)
	{

	l1.apply_filter(l1.get_filter(i),1);
	}

	for (gridEntity feature_map : l1.getFeatureMaps())
	{
		ImageInput i(feature_map);
		i.showImage();
	}






	// Go through the pixels;
	//for (int i = 0; i < pixelVals.size(); i++)
	//{
	//	for (int j = 0; j < pixelVals.size(); j++)
	//	{
	//		std::cout << pixelVals.at(i).at(j) << " ";

	//	std::cout << "\n\n\n\n";
	//}
}