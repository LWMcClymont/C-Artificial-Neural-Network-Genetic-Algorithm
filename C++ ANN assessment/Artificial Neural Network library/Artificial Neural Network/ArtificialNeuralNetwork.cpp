#include "ArtificialNeuralNetwork.h"

NeuralNetwork::NeuralNetwork(vector<unsigned int> structure)
{
	// Add a neuron for bias
	for (int i = 0; i < structure.size(); i++)
	{
		structure[i] += 1;
	}

	// Create weights with random values for each hidden layer 
	// Each weight is set between -1 & 1
	for (int i = 0; i < structure.size() - 1; i++)
	{
		m_hiddenLayers.push_back(Matrix(structure[i], structure[i + 1]));

		for (int r = 0; r < m_hiddenLayers.back().data.size(); r++)
		{
			for (int c = 0; c < m_hiddenLayers.back().data[r].size(); c++)
			{
				m_hiddenLayers.back().data[r][c] = (double)rand() / (double)RAND_MAX;
			}
		}
	}
}

void NeuralNetwork::FeedForward(Matrix & data)
{
	m_neuronLayers = vector<Matrix>();

	// Create a Matrix with the same dimensions as the data plus one extra for bias
	Matrix trainingData = Matrix(data.data.size(), data.data[0].size() + 1);

	// Set the bias
	for (int r = 0; r < trainingData.data.size(); r++)
	{
		trainingData[r][trainingData.data[0].size() - 1] = m_bias;
	}

	// Copy info from 'data' matrix into the new matrix
	for (int r = 0; r < trainingData.data.size(); r++)
	{
		for (int c = 0; c < trainingData.data[0].size() - 1; c++) // we don't want to iterate over bias
		{
			trainingData[r][c] = data[r][c];
		}
	}

	m_result = trainingData;

	// Pass our inputs through the hidden layers
	for (int l = 0; l < m_hiddenLayers.size(); l++)
	{
		m_result = m_result * m_hiddenLayers[l];

		// Apply activation function
		for (int r = 0; r < m_result.data.size(); r++)
		{
			for (int c = 0; c < m_result[0].size(); c++)
			{
				switch (m_activationFunction)
				{
					case SIGMOID: m_result[r][c] = Sigmoid(m_result[r][c]); break;
					case TANH: m_result[r][c] = TanH(m_result[r][c]); break;
					case RELU: m_result[r][c] = Relu(m_result[r][c]); break;
				}
			}

			// Reset the bias slot to the bias value
			m_result[r][m_result[0].size() - 1] = m_bias;
		}

		// Add result to neuron container
		m_neuronLayers.push_back(m_result);
	}
}

Matrix NeuralNetwork::GetResult()
{
	return m_result;
}

void NeuralNetwork::SetBias(double bias)
{
	m_bias = bias;
}

void NeuralNetwork::SetActivationFunction(int function)
{
	m_activationFunction = function;
}

double NeuralNetwork::Sigmoid(double & x)
{
	return 1 / (1 + (pow(2.718281828459045, -x)));
}

double NeuralNetwork::TanH(double & x)
{
	return tanh(x);
}

double NeuralNetwork::Relu(double & x)
{
	return max(0.0, x);
}

GANeuralNetwork::GANeuralNetwork(vector<unsigned int> structure, unsigned int populationSize, unsigned int maxGenerations)
{
	m_populationSize = populationSize;
	m_maxGenerations = maxGenerations;
	m_fittest = -1;

	for (int i = 0; i < populationSize; i++)
	{
		m_population.push_back(Individual(NeuralNetwork(structure)));
	}
}

bool GANeuralNetwork::Train(Data & data)
{
	m_fittest = 0;

	// Run through each NN in population and feed forward our data
	// Then work out the individual NN error
	for (int i = 0; i < m_population.size(); i++)
	{
		Individual & ind = m_population[i];

		ind.m_error = 0;

		// Feed out data through the network
		ind.m_net.FeedForward(data.input);
		Matrix & result = ind.m_net.GetResult();

		// Calculate the error -- RMS
		for (int r = 0; r < result.data.size(); r++)
		{
			for (int c = 0; c < result.data[r].size() - 1; c++)
			{
				double delta = (data.expectedOutput[r][c] - result[r][c]);
				ind.m_error += delta * delta;
			}
		}

		if (ind.m_error <= m_population[m_fittest].m_error)
		{
			m_fittest = i;
		}
	}

	int generationCount = 0;

	// GA -- termination is either if reasonable solution found or
	// We've reached our maximum generation count
	while (m_population[m_fittest].m_error > 0.01 & generationCount < m_maxGenerations)
	{
		// Select second fittest for crossover
		int secondFittest = 0;
		
		for (int i = 0; i < m_population.size(); i++)
		{
			if (i == m_fittest)
				continue;
			if (m_population[i].m_error <= m_population[secondFittest].m_error)
			{
				secondFittest = i;
			}
		}
		

		// Create child and initialize it with the fittest's weights
		Individual i = m_population[m_fittest];

		// Randomly select weights to crossover
		// Also mutate the them by a random amount
		for (int l = 0; l < i.m_net.m_hiddenLayers.size(); l++)
		{
			Matrix & m = i.m_net.m_hiddenLayers[l];

			for (int r = 0; r < m.data.size(); r++)
			{
				for (int c = 0; c < m.data[r].size(); c++)
				{
					// CrossOver
					bool useSecondFittestGenes = (rand() % 10) > 5 ? true : false;
					if (useSecondFittestGenes)
					{
						m[r][c] = m_population[secondFittest].m_net.m_hiddenLayers[l].data[r][c];
					}

					// Mutate
					float mutationRate = (double)rand() / (RAND_MAX + 1) * (0.5 - -0.5) - 0.5;
					m[r][c] += mutationRate;
				}
			}
		}

		i.m_error = 0;
		
		// Feed our data into the new child
		i.m_net.FeedForward(data.input);

		// Work out error of child
		Matrix & result = i.m_net.GetResult();
		for (int r = 0; r < result.data.size(); r++)
		{
			for (int c = 0; c < result.data[0].size() - 1; c++)
			{
				double delta = (data.expectedOutput[r][c] - result[r][c]);
				i.m_error += delta * delta;
			}
		}

		// If this child is better than a current individual in the population 
		// Then this child will replace it
		for (int ii = 0; ii < m_population.size(); ii++)
		{
			if (m_population[ii].m_error >= i.m_error)
			{
				m_population[ii] = i;
			}
		}

		generationCount++;
		//cout << "GA iteration: " << generationCount << endl;
		cout << "Cur fittest error: " << m_population[m_fittest].m_error << endl;
	}

	return true;
}


Individual GANeuralNetwork::GetFittestNN()
{
	if (m_fittest == -1)
	{
		cout << "No fittest has been selected." << endl;
	}

	return m_population[m_fittest];
}
