#pragma once
class MetropolisXOR
{
private:
	const int* TruthTable;
	int size;
	float CurrentWeights[16] = { 0 };
	float NewWeights[16] = { 0 };
	float Beta = 0.1f;
public:
	MetropolisXOR(const int* TruthTable, float* InitWeights);
	~MetropolisXOR();
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

