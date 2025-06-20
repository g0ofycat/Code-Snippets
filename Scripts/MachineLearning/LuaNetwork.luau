--!native
--!optimize 2

--[[ NOTES:
	
	- Each neuron is connected to every single one in the next layer and so on
	- Weights: The strength of the 	connections between each neuron
	- Biases: Makes the result more offset, allowing for (possibly) better responses
	- Neurons: A single neuron that makes up a whole neural network, each neural is fed a weighted sum from the previous layer, it's then added plus the bias. (WeightedSum + Bias)
	- Sigmoid: An activation for the neuron, it returns a value from 0-1. It allows networks non-linearity (Learning Complex Patterns)
	- Cost: Costs are useful since it provides the error margin through a number using the Output of a neural network and comparing it to the actual output given. It's used to refine the output and eventully get the right Output
	
	- Currently using: Xavier Initialization, ReLU, Softmax, Cross-Entropy to calculate cost, 2 feed forward network.
	- You need to flatten the MNIST Dataset and also export the biases and weights to get actually accurate results, I used Python (Tensorflow) for this.
	- Training is quite useless, just a misc function since training in Roblox is unreliable.
]]

local ActivationLib = require(script.ActivationLib)
local LuaNetwork = {}
LuaNetwork.__index = LuaNetwork

---------------------------------------------
-- // Helper Functions
---------------------------------------------

function CostCalculator(Outputs: {number}, Expected: {number}): number -- // The (average) amount of error between what my module predicts VS the actual value. (Uses Cross-Entropy)
	local sum = 0
	for i = 1, #Outputs do
		local pred = math.clamp(Outputs[i], 1e-7, 1 - 1e-7)
		sum += -Expected[i] * math.log(pred)
	end
	return sum / #Outputs
end

---------------------------------------------
-- // Propagations
---------------------------------------------

function LuaNetwork:ForwardPropagation(Data: {number}): ({number}, {number})
	local hiddenInputs = {}
	local hiddenOutputs = {}
	for j = 1, self.hiddenNodes do -- // Input to Hidden Layer
		local sum = 0
		for i = 1, self.inputNodes do
			sum += Data[i] * self.weightsIH[i][j]
		end
		hiddenInputs[j] = sum + self.biasH[j]
		hiddenOutputs[j] = ActivationLib.ReLU(hiddenInputs[j]) -- // ReLU for rest of Layers
	end

	local outputInputs = {}
	local outputs = {}
	for k = 1, self.outputNodes do -- // Hidden Layer to Output Layer
		local sum = 0
		for j = 1, self.hiddenNodes do
			sum += hiddenOutputs[j] * self.weightsHO[j][k]
		end
		outputInputs[k] = sum + self.biasO[k]
	end
	outputs = ActivationLib.Softmax(outputInputs) -- // Softmax for the outputs
	return hiddenOutputs, outputs
end

-- // Obviously this is just a helper function since BackPropagation is used in training. Do training on like python or something
function LuaNetwork:BackPropagation(Data: {number}, Target: {number}, HiddenOutputs: {number}, Outputs: {number}, LearningRate: number)
	local outputGradients = {}
	for k = 1, self.outputNodes do
		outputGradients[k] = Outputs[k] - Target[k]
		for j = 1, self.hiddenNodes do
			self.weightsHO[j][k] -= LearningRate * outputGradients[k] * HiddenOutputs[j]
		end
		self.biasO[k] -= LearningRate * outputGradients[k]
	end

	local hiddenGradients = {}
	for j = 1, self.hiddenNodes do
		local error = 0
		for k = 1, self.outputNodes do
			error += outputGradients[k] * self.weightsHO[j][k]
		end
		hiddenGradients[j] = error * ActivationLib.SigmoidDerivative(HiddenOutputs[j])
		for i = 1, self.inputNodes do
			self.weightsIH[i][j] -= LearningRate * hiddenGradients[j] * Data[i]
		end
		self.biasH[j] -= LearningRate * hiddenGradients[j]
	end
end

function LuaNetwork:LoadWeights(weights)
	self.weightsIH = weights.weightsIH
	self.weightsHO = weights.weightsHO
	self.biasH = weights.biasH
	self.biasO = weights.biasO
end

---------------------------------------------
-- // Constructor
---------------------------------------------

--[[
    :New() The actual constructor for the function, handles the weights and biases for the neurons.
    
    - InputNodes: The amount of nodes used for the input
    - HiddenNodes: The amount of nodes used for the hidden layers
    - OutputNodes: The amount of nodes used for the output
    - Seed: Optional parameter to set the seed of Random.new()
]]
function LuaNetwork.New(InputNodes: number, HiddenNodes: number, OutputNodes: number, Seed: number, PretrainedWeights: {number}?)
	if InputNodes <= 0 or HiddenNodes <= 0 or OutputNodes <= 0 then
		error("InputNodes, HiddenNodes, and OutputNodes must be positive")
	end

	local Random = Seed and Random.new(Seed) or Random.new()

	local self = setmetatable({
		inputNodes = InputNodes,
		hiddenNodes = HiddenNodes,
		outputNodes = OutputNodes,
		weightsIH = {}, -- // Input neurons to hidden neurons weights
		weightsHO = {}, -- // Hidden neurons to output neurons weights
		biasH = {}, -- // Biases for the Hidden Neurons
		biasO = {} -- // Biases for the Output Neurons
	}, LuaNetwork)

	if PretrainedWeights then -- // Using Python
		self:LoadWeights(PretrainedWeights)
	else
		for i = 1, InputNodes do -- // Init Weights and Biases for all tables (-1, 1), (Weights are initialized by Uniform Xavier)
			self.weightsIH[i] = {}
			for j = 1, HiddenNodes do
				local scaleIH = math.sqrt(6 / (InputNodes + HiddenNodes))
				self.weightsIH[i][j] = Random:NextNumber(-scaleIH, scaleIH)
			end
		end
		for j = 1, HiddenNodes do
			self.weightsHO[j] = {}
			for k = 1, OutputNodes do
				local scaleHO = math.sqrt(6 / (HiddenNodes + OutputNodes))
				self.weightsHO[j][k] = Random:NextNumber(-scaleHO, scaleHO)
			end
		end
		for j = 1, HiddenNodes do
			self.biasH[j] = Random:NextNumber(-0.1, 0.1)
		end
		for k = 1, OutputNodes do
			self.biasO[k] = Random:NextNumber(-0.1, 0.1)
		end
	end

	return self
end

--[[
	:Predict() Used to feed in data to the Neural Network for actual training
	
	- Data: A table of numbers for the neural network to learn from
]]
function LuaNetwork:Predict(Data: {number}): {number}
	if #Data ~= self.inputNodes then
		error(string.format("Predict: Data length is %d, expected %d input nodes", #Data, self.inputNodes))
	end
	local _, outputs = self:ForwardPropagation(Data)
	return outputs
end

--[[
	:Train() Used to train the data using Forward and Backpropagation. (Changes Weights and Biases, don't use if you already have pre determined weights)
	
	- Data: A table of numbers for the neural network to learn from
	- Target: A table that has the desired values you want the Neural Network to learn
	- LearningRate: How quickly the Neural Network adjusts it's parameters to learn
]]
function LuaNetwork:Train(Data: {number}, Target: {number}, LearningRate: number): (number, {number})
	if #Data ~= self.inputNodes then
		error(string.format("Train: Data length is %d, expected %d input nodes", #Data, self.inputNodes))
	end
	if #Target ~= self.outputNodes then
		error(string.format("Train: Target length is %d, expected %d output nodes", #Target, self.outputNodes))
	end
	if LearningRate < 0.0001 or LearningRate > 1 then
		warn(string.format("LearningRate %.4f is outside recommended range [0.0001, 1], clamping", LearningRate))
		LearningRate = math.clamp(LearningRate, 0.0001, 1)
	end

	local hiddenOutputs, outputs = self:ForwardPropagation(Data)
	local cost = CostCalculator(outputs, Target)
	self:BackPropagation(Data, Target, hiddenOutputs, outputs, LearningRate)
	return cost, outputs
end

return LuaNetwork
