#include "Network.h"

double Network::erreurMoyenneSmoothFactor = 100.0;

Network::Network(const std::vector<unsigned>& topology)	
{
	unsigned numLayers = topology.size();
	for (unsigned layerNb = 0; layerNb < numLayers; layerNb++)
	{
		std::cout << std::endl << "Added a Layer" << std::endl;
		// On ajoute une couche
		layers.push_back(Layer());
		unsigned nbOutputs = (layerNb == topology.size() - 1) ? 0 : topology[layerNb + 1];

		// Il faut ajouter chacun des neurones dans la couche
		for (unsigned neuronNb = 0; neuronNb <= topology[layerNb]; ++neuronNb)
		{
			layers.back().push_back(Neuron(nbOutputs, neuronNb));
			std::cout << "Made a Neuron" << std::endl;
		}

		// Bias a 1.0
		layers.back().back().setOutputVal(1.0);
	}
}

double Network::getRecentAverageError()
{
	return erreurMoyenne;
}

void Network::feedForward(const std::vector<double>& inputVals)
{
	assert(inputVals.size() == layers[0].size() - 1);

	// On assigne les premiers neurones
	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		layers[0][i].setOutputVal(inputVals[i]);
	}

	// On parcourt ensuite chaque couche 
	for (unsigned layerNb = 1; layerNb < layers.size(); ++layerNb)
	{
		Layer& previousLayer = layers[layerNb - 1];
		// On continue la propagation
		for (unsigned n = 0; n < layers[layerNb].size() - 1; ++n)
		{
			layers[layerNb][n].feedForward(previousLayer);
		}
	}
}

void Network::backProp(const std::vector<double>& targetVals)
{
	// Calcul de l'erreur du Network

	Layer& outputLayer = layers.back();
	erreur = 0.0;
	
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		erreur += delta * delta;
	}

	erreur /= outputLayer.size() - 1;
	erreur = sqrt(erreur);

	// Mesure de l'erreur moyenne

	erreurMoyenne = (erreurMoyenne * erreurMoyenneSmoothFactor + erreur) / (erreurMoyenneSmoothFactor + 1.0);

	// Calcul des gradients sur la couche de sortie

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calcul des gradients sur les couches intermediaires
	
	for (unsigned layerNb = layers.size() - 2; layerNb > 0; --layerNb)
	{
		Layer& hiddenLayer = layers[layerNb];
		Layer& nextLayer = layers[layerNb + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// On update les poids de toutes les couches

	for (unsigned layerNb = layers.size() - 1; layerNb > 0; --layerNb)
	{
		Layer& layer = layers[layerNb];
		Layer& prevLayer = layers[layerNb - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Network::getResults(std::vector<double>& resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		resultVals.push_back(layers.back()[n].getOutputVal());
	}
}
