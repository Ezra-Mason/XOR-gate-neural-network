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
	void Run();
	void LogResults();
private:
	/// <summary>
	/// Determine the error between the given weights and the truth table
	/// </summary>
	/// <param name="Weights"></param>
	/// <returns></returns>
	float GetGobalError(float* Weights);
	float SigmoidThreshold(float input);
};

