#include "ImageInput.hpp"
#include "ConvolutionLayers.hpp"

int main()
{
	
	ImageInput* img = new ImageInput("drawing.jpg", CV_8UC1);
	img->showImage();
	
	// Get matrixified pixel values for the images
	std::vector<std::vector<double>> pixelVals = img->getMatrixifiedPixelValues();



	// Start the first convolution Layer

	// Apply filters for 1st convlolution layer
	ConvolutionLayers l1(pixelVals);

	// ATTENTIONNNNNNNNNNNNNNNNNn -> 3 here is no. of filers, very bad code i know, fuck off
	for (int i = 0; i < 3; i++)
	{
		l1.apply_filter(l1.get_filter(i),1);
	}

	l1.activate_feature_maps_using_RELU();


	// Result
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