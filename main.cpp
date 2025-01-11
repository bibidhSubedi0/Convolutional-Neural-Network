#include "ImageInput.hpp"
#include "Matrix.hpp"
#include "ConvolutionLayers.hpp"
#include "DeepNetwork.hpp"
int main()
{

	ImageInput* img = new ImageInput("drawing.png", CV_8UC1);
	// img->showImage();

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

	// Activate feature maps
	for(gridEntity &f_map: l1.get_feature_map())
	{
		l1.activate_feature_map_using_RELU_universal(f_map);
	}


	//for (gridEntity feature_map : l1.get_feature_map())
	//{
	//	ImageInput i(feature_map);
	//	i.showImage();
	//}

	// Apply max pooling
	for (int i = 0; i < l1.get_feature_map().size(); i++)
	{
		gridEntity pMap = l1.apply_pooling_univeral(l1.get_feature_map().at(i), 2);
		l1.get_pool_map().push_back(pMap);
	}

	// print pooled maps
	//for (gridEntity pool : l1.get_pool_map())
	//{
	//	ImageInput i(pool);
	//	i.showImage();
	//}


	// ------------------------------------------------------------------
	// Put these final pool maps into the input channel for next layer
	// -----------------------------------------------------------------
	
	// the vector<gridEntity> pool_maps IS THE INPUT CHANNEL for next layer
	l1.get_input_channels() = l1.get_pool_map();


	// ------------------------------------------------------------------
	// Apply the 'volumetricEntity training_filters' designated to the second convolution layer and get the 'volumetricEntity ouput_features;'
	// -----------------------------------------------------------------


	std::vector<gridEntity> all_final_set_of_filter_maps;


	/*std::cout << "The filters to train are as follow : " << std::endl;
	for (auto f : l1.get_all_training_filter())
	{
		std::cout << "------------------------------------\n";
		std::cout << "#########################\n";
		
		for (auto sheet : f)
		{
			CNN_Matrix::Matrix::displayMatrix(sheet);
			std::cout << "#########################\n";
			
		}
		
		std::cout << "------------------------------------\n";
	}*/

	for (int filter_count = 0; filter_count < l1.get_all_training_filter().size(); filter_count++)
	{
		// first sheet of training filter to first sheet of input channel - will get 1st filter map
		// second sheet of traning filter to second sheet of input channel - will get 2nd filter map
		// . . . 
		// nth sheet of training filter to nth sheet of input channel - will get nth filter map
		
		std::vector<gridEntity> n_filter_maps;
		

		for (int i = 0; i < l1.get_input_channels().size(); i++) {
			// l1.get_all_training_filter().at(filter_count).at(0); // first sheet of filter
			// l1.get_input_channels().at(0); // first sheet of the input channel
			gridEntity nth_filter_map = l1.apply_filter_universal(l1.get_input_channels().at(i), l1.get_all_training_filter().at(filter_count).at(i),1);
			n_filter_maps.push_back(nth_filter_map);
		}

		
		// then sum these filter map -> 1st + 2nd + ... + nth  = summed_filter_map

		gridEntity sum_of_filters = CNN_Matrix::Matrix::sum_of_all_matrix_elements(n_filter_maps);

		all_final_set_of_filter_maps.push_back(sum_of_filters);
		// then go downard  to apply activation and normalaiztion
	}

	l1.get_output_feature_maps() = all_final_set_of_filter_maps;


	// ------------------------------------------------------------------
	// Apply activation and normalaization to the output features
	// I probably wont need as i am gon use RELU anyways
	// -----------------------------------------------------------------


	// Activate feature maps -> NOT WORKING
	for (gridEntity& f_map : l1.get_output_feature_maps())
	{
		l1.activate_feature_map_using_SIGMOID(f_map);
		ImageInput i(f_map);
		// i.showImage();
	}





	// ------------------------------------------------------------------
	// Apply pooling to the output features
	// -----------------------------------------------------------------
	
	// Apply max pooling
	for (int i = 0; i < l1.get_output_feature_maps().size(); i++)
	{
		gridEntity pMap = l1.apply_pooling_univeral(l1.get_output_feature_maps().at(i), 2);
		l1.get_final_pool_maps().push_back(pMap);
	}

	/*for (gridEntity pool : l1.get_final_pool_maps())
	{
		ImageInput i(pool);
		i.showImage();
	}*/



	// ------------------------------------------------------------------
	// Flatten the pooled layer
	// -----------------------------------------------------------------
	std::vector<double> flatVec;
	for (const auto& matrix : l1.get_final_pool_maps()) {
		for (const auto& row : matrix) {
			flatVec.insert(flatVec.end(), row.begin(), row.end());
		}
	}

	for (auto x : flatVec)
	{
		std::cout << x << std::endl;
	}



	// ------------------------------------------------------------------
	// Feed into deep neural network 
	// -----------------------------------------------------------------


	vector<double> inputs = flatVec ;
	vector<double> target = { 0,0,0,0,0,1,0,0,0,0 };
	double learning_rates =  0.01 ;// { 0.01, 0.1, , 1 };
	vector<int> topologies = { (int)inputs.size(),8,(int)target.size()} ; // { {4, 8, 4}, { 4,8,16,8,4 },  };

	DeepNetwork* Net = new DeepNetwork(topologies, learning_rates);
	Net->setCurrentInput(inputs);
	Net->setTarget(target);

	Net->forwardPropogation();

	Net->setErrors();

	std::cout << "Error is : " << Net->getGlobalError();


}


