#pragma once
#include "Matrix.h"
#include <vector>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <functional>

using std::vector;

#define SIGMOID 0
#define TANH 1
#define RELU 2

class Data
{
public:
	Matrix input = Matrix(0, 0);
	Matrix expectedOutput = Matrix(0, 0);

	Data(Matrix inputMatrix, Matrix expectedOutputMatrix)
	{
		input = inputMatrix;
		expectedOutput = expectedOutputMatrix;
	}
};

class NeuralNetwork
{
public:
	NeuralNetwork() {};
	NeuralNetwork(vector<unsigned int> structure);
	void FeedForward(Matrix & data);
	Matrix GetResult();
	void SetBias(double bias);
	vector<Matrix> m_hiddenLayers;
	vector<Matrix> m_neuronLayers;
	void SetActivationFunction(int function);

private:
	Matrix m_result = Matrix(0, 0);
	double m_bias = 1;
	double m_error; 

	// Activation function
	double Sigmoid(double & x);
	double TanH(double & x);
	double Relu(double & x);

	int m_activationFunction = 0;
};

struct Individual
{
	Individual(NeuralNetwork nn)
	{
		m_net = nn;
		m_error = 0;
	};
	NeuralNetwork m_net;
	double m_error;
};


class GANeuralNetwork
{
public:
	GANeuralNetwork(vector<unsigned int> structure, unsigned int populationSize, unsigned int maxGenerations);
	bool Train(Data & data);
	Individual GetFittestNN();
	vector<Individual> m_population;

private:
	int m_populationSize;
	int m_maxGenerations;
	int m_fittest;
};

