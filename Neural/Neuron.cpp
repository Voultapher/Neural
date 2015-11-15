#include "Neuron.h"
#include "Random.h"

#include <cstdio>

Neuron::Neuron(unsigned int numOutputs, std::size_t id)
{
	//printf("Debug a Neuron was added\n");
	_id = id;
	_outputWeights.reserve(numOutputs);
	for (std::size_t c = 0; c < numOutputs; ++c){
		_outputWeights.push_back(Connection());
		//printf("Debug Added a outputWeight number %d\n", c);
		_outputWeights.back().weight = randomWeight();
	}
}


Neuron::~Neuron()
{
}

void Neuron::setOutputValue(double outputVal){
	_outputVal = outputVal;
}

void Neuron::setGradient(double gradient){
	_gradient = gradient;
}

double Neuron::getConnectionWeight(std::size_t id) const{
	return _outputWeights[id].weight;
}

double Neuron::getConnectionDeltaWeight(std::size_t id) const{
	return _outputWeights[id].deltaWeight;
}

void Neuron::updateConnection(std::size_t id, double weight){
	_outputWeights[id].weight += weight;
	_outputWeights[id].deltaWeight = weight;
}