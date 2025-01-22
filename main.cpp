#include "ImageInput.hpp"
#include "Matrix.hpp"
#include "ConvolutionLayers.hpp"
#include "DeepNetwork.hpp"
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
		gridEntity f_map = l1.apply_filter_universal(l1.get_raw_input_image(), l1.get_all_predefined_filter().at(filter_count), 1);
		l1.get_feature_map().push_back(f_map);
	}

	// Activate feature maps
	for (gridEntity& f_map : l1.get_feature_map())
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



	for(int epoch=0;epoch <10;epoch++){
		std::cout << "Training Epoch" <<epoch<< std::endl;
		// ------------------------------------------------------------------
		// Apply the 'volumetricEntity training_filters' designated to the second convolution layer and get the 'volumetricEntity ouput_features;'
		// -----------------------------------------------------------------
		std::vector<gridEntity> all_final_set_of_filter_maps;
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
				gridEntity nth_filter_map = l1.apply_filter_universal(l1.get_input_channels().at(i), l1.get_all_training_filter().at(filter_count).at(i), 1);
				n_filter_maps.push_back(nth_filter_map);
			}


			// then sum these filter map -> 1st + 2nd + ... + nth  = summed_filter_map

			gridEntity sum_of_filters = CNN_Matrix::Matrix::sum_of_all_matrix_elements(n_filter_maps);

			all_final_set_of_filter_maps.push_back(sum_of_filters);
			// then go downard  to apply activation and normalaiztion
		}
		l1.get_output_feature_maps() = all_final_set_of_filter_maps;
		// ------------------------------------------------------------------
		// Apply pooling to the output features
		// -----------------------------------------------------------------
		for (int i = 0; i < l1.get_output_feature_maps().size(); i++)
		{
			gridEntity pMap = l1.apply_pooling_univeral(l1.get_output_feature_maps().at(i), 2);
			l1.get_final_pool_maps().push_back(pMap);
		}
		// ------------------------------------------------------------------
		// Flatten the pooled layer
		// -----------------------------------------------------------------
		int chanels = l1.get_final_pool_maps().size(); // no. of filters
		int filt_height = l1.get_final_pool_maps().at(0).size(); // rows
		int filt_width = l1.get_final_pool_maps().at(0).at(0).size();
		std::vector<double> flatVec;
		for (const auto& matrix : l1.get_final_pool_maps()) {
			for (const auto& row : matrix) {
				flatVec.insert(flatVec.end(), row.begin(), row.end());
			}
		}
		// ------------------------------------------------------------------
		// Feed into deep neural network 
		// -----------------------------------------------------------------
		vector<double> inputs = flatVec;
		vector<double> target = { 0,1 };
		double learning_rates = 0.01;
		vector<int> topologies = { (int)inputs.size(),400,(int)target.size() }; 
		DeepNetwork* Net = new DeepNetwork(topologies, learning_rates);
		Net->setCurrentInput(inputs);
		Net->setTarget(target);
		Net->forwardPropogation();
		Net->setErrors();
		Net->gardientComputation();
		std::vector<GeneralMatrix::Matrix*> GradientMatrices = Net->GetGradientMatrices();
		Net->updateWeights();
		//-----------------------------------------------------------------------------------------------------
		// Reshape the gradients in kernal form i.e. for not channels '3' ota widthxheight '11x11' ko array
		//-----------------------------------------------------------------------------------------------------
		GeneralMatrix::Matrix* reqGrads = new GeneralMatrix::Matrix(1, flatVec.size(), false); 
		GeneralMatrix::Matrix* last_gradiet = GradientMatrices.at(GradientMatrices.size() - 1);
		GeneralMatrix::Matrix* tranposedWeightMatrices = Net->GetWeightMatrices().at(0)->tranpose();
		reqGrads = *last_gradiet * tranposedWeightMatrices;
		volumetricEntity poolGradients; 
		int last_pos = 0;
		int next_pos = filt_width * filt_width - 1;
		gridEntity temp(filt_width);
		int col = 0;
		for (int ch = 0; ch < chanels; ch++)
		{
			gridEntity temp;
			for (int i = 0; i < filt_width; i++)
			{
				std::vector<double> te;
				for (int j = 0; j < filt_height; j++)
				{
					double x = reqGrads->getVal(0, col);
					// std::cout << x << std::endl;
					te.push_back(x);
					col++;
				}
				temp.push_back(te);
			}
			poolGradients.push_back(temp);
		}
		
		//--------------------------------------------------------------------------------------------------------
		// Unpool the gardients from their pooled map to get the gradients to the filter maps
		//--------------------------------------------------------------------------------------------------------
		std::vector<gridEntity> filterMapGradients(l1.get_all_training_filter().size(),gridEntity(l1.get_output_feature_maps().at(0).size(), std::vector<double>(l1.get_output_feature_maps().at(0).at(0).size(), 0.0)));
		volumetricEntity unpooledGradients;
		for (int ch = 0; ch < chanels; ch++)
		{
			std::cout << "  " << ch << std::endl;
			gridEntity unpooled_map = l1.unpool_without_indices(poolGradients[ch], all_final_set_of_filter_maps[ch], 2,2,2);
			unpooledGradients.push_back(unpooled_map);
		}
		//-----------------------------------------------------------------------------------------------------
		// Get the gradients for filters from the gradients to the filtermaps we just obtained
		// -----------------------------------------------------------------------------------------------------
		std::vector<gridEntity> inputChannels = l1.get_input_channels();
		std::vector<volumetricEntity> filterGradients = l1.compute_filter_gradients(inputChannels,unpooledGradients,1);
		//-----------------------------------------------------------------------------------------------------
		// Update the filters based on the computed Gradients
		//-----------------------------------------------------------------------------------------------------
		double learningRate = 0.01; // Example learning rate
		l1.update_filters_with_gradients(l1.get_all_training_filter(),filterGradients,learningRate);

		std::cout << "Network info: " << std::endl;
		std::cout << "No. of feature maps from initial CL : "<<l1.get_feature_map().size() << std::endl;
		std::cout << "No. of traniable filters : " << l1.get_all_training_filter().size() << std::endl;
		std::cout << "Dimention of trainable filter :" <<l1.get_all_training_filter().at(0).size() << "x"<< l1.get_all_training_filter().at(0).at(0).size() <<std::endl;
		std::cout << "No. of filters maps after application of trainable filters : " <<l1.get_output_feature_maps().size()<< std::endl;
		std::cout << "Dimention of filter map after application of trainable filters : " << l1.get_output_feature_maps().at(0).size()<<"x"<< l1.get_output_feature_maps().at(0).at(0).size() << std::endl;
		std::cout << "No. of pooled maps for those filter maps :" << l1.get_final_pool_maps().size() << std::endl;
		std::cout << "Dimention of pooled map :" <<l1.get_final_pool_maps().at(0).size()<<"x"<< l1.get_final_pool_maps().at(0).at(0).size() << std::endl;
	}

	
}


