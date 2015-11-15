#include "Net.h"
#include "Errors.h"

#include <math.h>


Net::Net(const std::vector<unsigned int> topology, double upperBound):
_eta(0.15),
_alpha(0.5),
_smoothingFactor(100.0)

{ // most of the ugly code is here, so that you have less ugly code later
	_scalingFactor = upperBound;

	// front and end of topology are split into seperate Layers for ease of use and type safety later on
	_inputLayer = Layer();
	_inputLayer.reserve(topology.front() + 1);
	for (std::size_t i = 0; i <= topology.front(); ++i){ // <= adds the hidden bias Neuron that the user wont see
		_inputLayer.push_back(Neuron(topology[1], i)); // add the Neurons and pass in the number of outputs it will have
	}
	_inputLayer.back().setOutputValue(1.0); // force the bias neuron to be 1.0

	_hiddenLayers.reserve(topology.size() - 2);
	for (std::size_t layerNum = 1; layerNum < topology.size() - 1; layerNum++){
		_hiddenLayers.push_back(Layer());
		_hiddenLayers.back().reserve(topology[layerNum] + 1);

		for (std::size_t i = 0; i <= topology[layerNum]; ++i){ // <= adds the hidden bias Neuron that the user wont see
			_hiddenLayers.back().push_back(Neuron(topology[layerNum + 1], i)); // add the Neurons and pass in the number of outputs it will have +1 for the hidden neuron
		}
		_hiddenLayers.back().back().setOutputValue(1.0); // force the bias neuron to be 1.0
	}
	_outputLayer = Layer();
	_outputLayer.reserve(topology.back() + 1);
	for (std::size_t i = 0; i <= topology.back(); ++i){
		_outputLayer.push_back(Neuron(0, i)); // add the last Neurons with a bias neuron and with 0 outputs
	}
	_outputLayer.back().setOutputValue(1.0);

	_allLayers.push_back(&_inputLayer); // construct the _allLayers vector, in some cases its easier to use this rather than the split up parts
	for (auto& hiddenLayer : _hiddenLayers){
		_allLayers.push_back(&hiddenLayer);
	}
	_allLayers.push_back(&_outputLayer);
}


Net::~Net()
{
}

bool Net::feedForward(const std::vector<double> inputVals){
	if (inputVals.size() == _inputLayer.size() - 1){ // -1 as the input layer has a hidden neuron
		for (std::size_t i = 0; i < inputVals.size(); ++i){
			_inputLayer[i].setOutputValue(inputVals[i] / _scalingFactor);
		}

		Layer* prevLayer = &_inputLayer;
		for (std::size_t layerNum = 1; layerNum < _allLayers.size(); ++layerNum){
			prevLayer = _allLayers[layerNum - 1];
			for (std::size_t n = 0; n < _allLayers[layerNum]->size() - 1; ++n){
				double sum = 0.0;
				for (auto& prevNeuron : (*prevLayer)){
					int id = (*_allLayers[layerNum])[n].getId();
					sum += prevNeuron.getOutputValue() * prevNeuron.getConnectionWeight(id);
				}
				//printf("Debug sum: %.5f, tranfer %.5f\n5", sum, transferFunction(sum));
				(*_allLayers[layerNum])[n].setOutputValue(transferFunction(sum));
			}
		}
		return true;
	}

	return false; // failed to feed data due to incorrect set Size
}

bool Net::backPropagate(const std::vector<double> targetVals){
	if (_outputLayer.size() - 1 == targetVals.size()){
		_error = 0.0;

		for (std::size_t n = 0; n < _outputLayer.size() - 1; ++n){
			double delta = targetVals[n] - _outputLayer[n].getOutputValue();
			_error += delta * delta;
		}
		_error /= _outputLayer.size() - 1;
		_error = sqrt(_error); // RMS

		_recentAverageError = (_recentAverageError * _smoothingFactor + _error) / (_smoothingFactor + 1.0);

		for (std::size_t n = 0; n < _outputLayer.size() - 1; ++n){ // calculate output layer gradient
			calcOutputGradients(_outputLayer[n], targetVals[n] / _scalingFactor);
		}
		Layer* deepLayer = &_outputLayer;
		for (std::vector<Layer>::reverse_iterator hiddenLayer = _hiddenLayers.rbegin(); hiddenLayer != _hiddenLayers.rend(); ++hiddenLayer){ // calculate hidden layer gradient in reverse order
			for (std::size_t n = 0; n < hiddenLayer->size(); ++n){
				calcHiddenGradients((*hiddenLayer)[n], *deepLayer);
			}
			deepLayer = &(*hiddenLayer);
		}

		deepLayer = &_outputLayer;
		for (std::size_t layerNum = _allLayers.size() - 1; layerNum > 0; --layerNum){ // update hidden layer gradient in reverse order
			for (std::size_t n = 0; n < deepLayer->size() - 1; ++n){ // as he ignores the the bias neuron
				updateInputWeights((*deepLayer)[n], _allLayers[layerNum - 1]);
			}
			deepLayer = _allLayers[layerNum - 1];
		}
		return true;
	}

	return false;
}

void Net::getResults(std::vector<double>& resultVals) const{
	resultVals.clear();
	resultVals.reserve(_outputLayer.size());

	for (std::size_t n = 0; n < _outputLayer.size() - 1; ++n){
		resultVals.push_back(_outputLayer[n].getOutputValue());
	}
}

void Net::calcOutputGradients(Neuron& neuron, double targetVal){
	double delta = targetVal - neuron.getOutputValue();
	neuron.setGradient(delta * transferFunctionDerivative(neuron.getOutputValue()));
	//printf("Debug: gradient: %f\n", neuron.getGradient());
}
void Net::calcHiddenGradients(Neuron& neuron, const Layer& prevLayer){
	double weightSum = calcWeightSum(neuron, prevLayer);
	neuron.setGradient(weightSum * transferFunctionDerivative(neuron.getOutputValue()));
	//printf("Debug: gradient: %f\n", neuron.getGradient());
}

void Net::updateInputWeights(Neuron& neuron, Layer* prevLayer){
	for (auto& prevNeuron : *prevLayer){
		//printf("Debug neuron.getId() : %d prevNeuron.getConnectionSize(): %d\n", neuron.getId(), prevNeuron.getConnectionSize());
		double oldDeltaWeight = prevNeuron.getConnectionDeltaWeight(neuron.getId());
		//newDeltaWeight: Individual input, magnified by the gradient and train rate + momentum = a fraction of the previous delta witht;
		double newDeltaWeight = _eta * prevNeuron.getOutputValue() * neuron.getGradient() + _alpha * oldDeltaWeight;
		//printf("Debug: oldDeltaWeight: %.5f, newDeltaWeight: %.5f\n", oldDeltaWeight, newDeltaWeight);

		prevNeuron.updateConnection(neuron.getId(), newDeltaWeight);
	}
}

double Net::transferFunction(double x){
	return tanh(x);
}

double Net::transferFunctionDerivative(double x){
	return (1.0 - x * x); // a good enough approximation for our usecase
}
double Net::calcWeightSum(const Neuron& neuron, const Layer& prevLayer){
	double sum = 0.0; // sum our contributions of the errors at the nodes we feed

	for (std::size_t n = 0; n < prevLayer.size() -1; ++n){
		sum += neuron.getConnectionWeight(n) * prevLayer[n].getGradient();
	}
	return sum;
}