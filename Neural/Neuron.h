#pragma once

#include <vector>
#include <random>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned int numOutputs, std::size_t id);
	~Neuron();

	void feedForward();

	void setOutputValue(double outputVal);
	void setGradient(double gradient);
	void updateConnection(std::size_t id, double weight);

	std::size_t getId() const { return _id; }
	double getOutputValue() const { return _outputVal; };
	double getGradient() const { return _gradient; };
	double getConnectionWeight(std::size_t id) const;
	double getConnectionDeltaWeight(std::size_t id) const;

	std::size_t getConnectionSize() const{ return _outputWeights.size(); }

private:
	std::size_t _id;
	double _outputVal;
	double _gradient;
	std::vector<Connection> _outputWeights;
};