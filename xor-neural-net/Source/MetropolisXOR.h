#pragma once
#include <vector>

class MetropolisXOR
{
private:
	const int* TruthTable;
	const std::vector<std::vector<int>>& Inputs;
	float CurrentWeights[16] = { 0 };
	float NewWeights[16] = { 0 };
	const int InputNodeCount;
	const int HiddenNodeCount;
	float Beta = 0.1f;
public:
	MetropolisXOR(
		const int InputNodeCount, 
		const std::vector<std::vector<int>>& Inputs,
		const int* TruthTable, 
		const int HiddenNodeCount, 
		float* InitWeights);
	/// <summary>
	/// Run the learning algorithm on the network
	/// </summary>
	void Run();
	/// <summary>
	/// Log the output of the networks learning to the console
	/// </summary>
	void LogResults();
private:
	/// <summary>
	/// Feed the given inputs through the network with the given weights
	/// </summary>
	/// <param name="Weights">The weights of the connections in the network</param>
	/// <param name="Inputs">Input values into the network</param>
	/// <param name="HiddenOutputs">Buffer for the values of the hidden node outputs</param>
	/// <returns>The value of the output node of the network</returns>
	float FeedForward(float* Weights, const std::vector<int>& Inputs, std::vector<float>& HiddenOutputs);
	/// <summary>
	/// Determine the error between the given weights and the truth table
	/// </summary>
	/// <param name="Weights">The weights of the connections in the network</param>
	/// <returns>The global error of a network with the given weights</returns>
	float GetGobalError(float* Weights);
	/// <summary>
	/// Threshold function to add nonlinearity to the activation of the nodes
	/// </summary>
	/// <param name="input">activation value</param>
	/// <returns>The nonlinear output</returns>
	float SigmoidThreshold(float input);
};

