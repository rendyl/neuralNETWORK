#pragma once
#include <vector>
#include <iostream>
#include <assert.h>

#include "Neuron.h"

typedef std::vector<Neuron> Layer;

class Network
{
	public :

		Network(const std::vector<unsigned> & topology);
		void feedForward(const std::vector<double> & inputValues);
		void backProp(const std::vector<double>& targetValues);
		void getResults(std::vector<double>& resultVals) const;

		double getRecentAverageError();

	private :

		double erreur;
		double erreurMoyenne;
		static double erreurMoyenneSmoothFactor;
		std::vector<Layer> layers; // m_layers[layerNb][neuronNb]
};

