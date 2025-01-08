#include "ImageInput.hpp"
#include "ConvolutionLayers.hpp"

int main()
{
	ImageInput* img = new ImageInput("drawing.png", CV_8UC1);
	img->showImage();

	// Get matrixified pixel values for the images
	std::vector<std::vector<double>> pixelVals = img->getMatrixifiedPixelValues();

	/*-----------------------------New Way-------------------------------------*/
	// Start the first convolution Layer
	ConvolutionLayers l1(pixelVals);

	for (int filter_count = 0; filter_count < l1.get_all_predefined_filter().size(); filter_count++)
	{
		gridEntity f_map=  l1.apply_filter_universal(l1.get_raw_input_image(), l1.get_all_predefined_filter().at(filter_count), 1);
		l1.get_feature_map().push_back(f_map);
	}

	// Activate feature maps -> NOT WORKING
	for(gridEntity &f_map: l1.get_feature_map())
	{
		l1.activate_feature_map_using_RELU_universal(f_map);
	}

	// May need to verify activation

	// Normaile the feature map
	for (gridEntity &f_map : l1.get_feature_map())
	{
		l1.apply_normalaization_universal(f_map);
	}

	for (gridEntity feature_map : l1.get_feature_map())
	{
		ImageInput i(feature_map);
		i.showImage();
	}

	// Apply max pooling
	for (int i = 0; i < l1.get_feature_map().size(); i++)
	{
		gridEntity pMap = l1.apply_pooling_univeral(l1.get_feature_map().at(i), 2);
		l1.get_pool_map().push_back(pMap);
	}

	// print pooled maps
	for (gridEntity pool : l1.get_pool_map())
	{
		ImageInput i(pool);
		i.showImage();
	}


	// ------------------------------------------------------------------
	// Put these final pool maps into the input channel for next layer
	// -----------------------------------------------------------------




	// ------------------------------------------------------------------
	// Apply the 'volumetricEntity training_filters' designated to the second convolution layer and get the 'volumetricEntity ouput_features;'
	// -----------------------------------------------------------------



	// ------------------------------------------------------------------
	// Apply activation and normalaization to the output features
	// -----------------------------------------------------------------



	// ------------------------------------------------------------------
	// Apply pooling to the output features
	// -----------------------------------------------------------------
	



	// ------------------------------------------------------------------
	// Flatten the pooled layer
	// -----------------------------------------------------------------





	// ------------------------------------------------------------------
	// Feed into deep neural network 
	// -----------------------------------------------------------------
}


