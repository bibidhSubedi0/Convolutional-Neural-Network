#include "ImageInput.hpp"
#include "ConvolutionLayers.hpp"

int main()
{
	ImageInput* img = new ImageInput("drawing.png", CV_8UC1);
	img->showImage();

	// Get matrixified pixel values for the images
	std::vector<std::vector<double>> pixelVals = img->getMatrixifiedPixelValues();



	// Start the first convolution Layer

	// Apply filters for 1st convlolution layer
	ConvolutionLayers l1(pixelVals);

	// ATTENTIONNNNNNNNNNNNNNNNNn -> 3 here is no. of filers, very bad code i know, fuck off
	for (int i = 0; i < 1; i++)
	{
		l1.apply_filter(l1.get_filter(i), 1);
	}



	l1.activate_feature_maps_using_RELU();


	// SO it seems activation is working but i also need to normaize the data to get pixel value between 0 and 255
	l1.apply_normalaization();



	// Result
	for (gridEntity feature_map : l1.getFeatureMaps())
	{
		ImageInput i(feature_map);
		i.showImage();
	}


	// Apply max pooling
	for (int i = 0; i < 1; i++)
	{
		l1.apply_pooling(l1.getFeatureMaps().at(i), 2);
	}

	// print pooled maps
	for (gridEntity pool : l1.getPoolMaps())
	{
		ImageInput i(pool);
		i.showImage();
	}



	
}

/*


	
	gridEntity nums = {
	{1, 2, 3, 4, 5},
	{6, 7, 8, 9, 10},
	{11, 12, 13, 14, 15},
	{16, 17, 18, 19, 20},
	{21, 22, 23, 24, 25}
	};

	ConvolutionLayers l1(nums);
	l1.apply_pooling(nums,2);


	for (int i = 0; i < l1.getPoolMaps().at(0).size(); i++)
	{
		for (int j = 0; j < l1.getPoolMaps().at(0).at(0).size(); j++)
		{
			std::cout << l1.getPoolMaps().at(0).at(i).at(j) << " ";
		}
		std::cout<<"\n";
	}

*/