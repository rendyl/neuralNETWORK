#include "Neuron.h"

static double randomWeight(void)
{
	return rand() / double(RAND_MAX);
}

static double fctTransfert(double x)
{
	// tanh
	return tanh(x);
}

double Neuron::eta = 0.3;
double Neuron::alpha = 0.5;

double Neuron::sumDOW(const std::vector<Neuron>& nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

static double fctTransfertDerivee(double x)
{
	return (1 - x * x);
}

Neuron::Neuron(unsigned nbOutputs, int index)
{
	myIndex = index;

	for (unsigned c = 0; c < nbOutputs; ++c)
	{
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}
}

double Neuron::getOutputVal() const
{
	return outputVal;
}

void Neuron::setOutputVal(double value)
{
	outputVal = value;
}

void Neuron::feedForward(const std::vector<Neuron>& prevLayer)
{
	double sum = 0.0;

	// On fait la sommes des f(iWi)
	for (unsigned n = 0; n < prevLayer.size(); ++n) 
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[myIndex].weight;
	}

	outputVal = fctTransfert(sum);
}

void Neuron::calcOutputGradients(const double & targetVal)
{
	double delta = targetVal - outputVal;
	gradient = delta * fctTransfertDerivee(outputVal);
}

void Neuron::calcHiddenGradients(const std::vector<Neuron>& nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * fctTransfertDerivee(outputVal);
}

void Neuron::updateInputWeights(std::vector<Neuron>& prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;
		double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;

		neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[myIndex].weight += newDeltaWeight;
	}
}