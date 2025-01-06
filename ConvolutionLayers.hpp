#pragma once
#include "all_includes.hpp"


class ConvolutionLayers
{
	// The raw image we get from input layer
	gridEntity raw_image;

	// All the filters we need to apply to the main image
	// I can extract the no of filters, filter dimentions from this data
	// Do note that, the filers are 2d filters as grid entity is a (2D) Matrix
	std::vector<gridEntity> filters;

	// The results after appying the filters to the main image
	// no. of filters = no. of feature_map
	std::vector<gridEntity> feature_maps;


	// Pool maps or smth, idk what they are called
	std::vector<gridEntity> pool_maps;

public:
	ConvolutionLayers(gridEntity);

	void apply_relu_to_filters();
	void apply_sigmoid_to_filters();

	std::vector<gridEntity> getFeatureMaps();
	std::vector<gridEntity> getPoolMaps();

	gridEntity get_filter(int);

	// gets an un-activated feature map from a filter
	// takes a filter and stride
	void apply_filter(gridEntity,int);

	// apply the activation for each feature map
	void activate_feature_maps_using_RELU();

	// apply normaization to the feature maps
	void apply_normalaization();

	// gets a pool map from the activaed feature maps
	// takes an activated feature map and stride
	void apply_pooling(gridEntity, int);


};

