#pragma once

#include <vector>

#include "Neuron.h"

class Net
{
public:
	Net(const std::vector<unsigned int> topology, double upperBound);
	~Net();

	bool feedForward(const std::vector<double> inputVals); // bool as the input may not be suitable for the current Net
	bool backPropagate(const std::vector<double> targetVals);

	double getScalingFactor() const { return _scalingFactor; }
	void getResults(std::vector<double>& resultVals) const;

private:
	void calcOutputGradients(Neuron& neuron, double targetVal);
	void calcHiddenGradients(Neuron& neuron, const Layer& prevLayer);
	void updateInputWeights(Neuron& neuron, Layer* prevLayer);

	double transferFunction(double x);
	double transferFunctionDerivative(double x);
	double calcWeightSum(const Neuron& neuron, const Layer& prevLayer);

	double _error;
	double _recentAverageError;
	Layer _inputLayer;
	Layer _outputLayer;
	std::vector<Layer> _hiddenLayers;
	std::vector<Layer*> _allLayers;

	double _scalingFactor;
	double _eta; // modifiable constants
	double _alpha;
	double _smoothingFactor;
};

