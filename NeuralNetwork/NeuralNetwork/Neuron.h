#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
	public :

		Neuron(unsigned nbOutputs, int myIndex);
		void setOutputVal(double value);
		double getOutputVal() const;
		void feedForward(const std::vector<Neuron>& prevLayer);

		double sumDOW(const std::vector<Neuron>& nextLayer) const;

		void calcOutputGradients(const double & targetVal);
		void calcHiddenGradients(const std::vector<Neuron>& nextLayer);
		void updateInputWeights(std::vector<Neuron>& prevLayer);

	private :

		static double eta; // learning rate / 0.0 : slow learner / 0.2 : medium learner / 1.0 : reckless learner
		static double alpha; // momentum / 0.0 : no momentum / 0.5 moderate momemtum

		int myIndex;
		double gradient;
		double outputVal;
		std::vector<Connection> outputWeights;
};

