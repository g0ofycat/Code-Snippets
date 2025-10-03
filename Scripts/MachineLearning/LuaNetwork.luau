--!native
--!optimize 2

--[[ NOTES:
	
	- Each neuron is connected to every single one in the next layer and so on
	- Weights: The strength of the connections between each neuron
	- Biases: Makes the result more offset, allowing for (possibly) better responses
	- Neurons: A single neuron that makes up a whole neural network, each neuron is fed a weighted sum from the previous layer, it's then added plus the bias. (WeightedSum + Bias)
	- Sigmoid: An activation for the neuron, it returns a value from 0-1. It allows networks non-linearity (Learning Complex Patterns)
	- Cost: Costs are useful since it provides the error margin through a number using the Output of a neural network and comparing it to the actual output given. It's used to refine the output and eventually get the right Output
	
	- Currently using: Xavier Initialization, ReLU, Softmax, Cross-Entropy to calculate cost, 2 feed forward network.
	- You need to flatten the MNIST Dataset and also export the biases and weights to get actually accurate results, I used Python (Tensorflow) for this.
	- Training is quite useless, just a misc function since training in Roblox is unreliable.
]]

-- ========================
-- // IMPORTS
-- ========================

local ActivationLib = require(script.ActivationLib)

-- ========================
-- // MAIN MODULE
-- ========================

local LuaNetwork = {}
LuaNetwork.__index = LuaNetwork

-- ========================
-- // HELPERS
-- ========================

-- CostCalculator(): Calculates the average error between predictions and expected values (Cross-Entropy).
-- @param Outputs: Predicted values
-- @param Expected: Expected values
-- @return cost: The average error
local function CostCalculator(Outputs: {number}, Expected: {number}): number
	local sum = 0
	for i = 1, #Outputs do
		local pred = math.clamp(Outputs[i], 1e-7, 1 - 1e-7)
		sum += -Expected[i] * math.log(pred)
	end
	return sum / #Outputs
end

-- ========================
-- // PROPAGATION
-- ========================

-- Propagates input data through the network.
-- @param Data: Input values
-- @return hiddenOutputs: Hidden layer outputs
-- @return outputs: Final outputs
function LuaNetwork:ForwardPropagation(Data: {number}): ({number}, {number})
	local hiddenInputs, hiddenOutputs = {}, {}

	for j = 1, self.hiddenNodes do -- // Input -> Hidden Layer
		local sum = 0
		for i = 1, self.inputNodes do
			sum += Data[i] * self.weightsIH[i][j]
		end
		hiddenInputs[j] = sum + self.biasH[j]
		hiddenOutputs[j] = ActivationLib.ReLU(hiddenInputs[j])
	end

	local outputInputs = {}
	for k = 1, self.outputNodes do -- // Hidden Layer -> Output Layer
		local sum = 0
		for j = 1, self.hiddenNodes do
			sum += hiddenOutputs[j] * self.weightsHO[j][k]
		end
		outputInputs[k] = sum + self.biasO[k]
	end

	local outputs = ActivationLib.Softmax(outputInputs)
	return hiddenOutputs, outputs
end

-- BackPropagation(): Adjusts weights/biases based on error (used in training).
-- @param Data: Input data
-- @param Target: Expected outputs
-- @param HiddenOutputs: Hidden layer outputs
-- @param Outputs: Final outputs
-- @param LearningRate: Adjustment rate
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

-- LoadWeights(): Loads pre-trained weights and biases.
-- @param weights table Table containing weightsIH, weightsHO, biasH, biasO
function LuaNetwork:LoadWeights(weights)
	self.weightsIH = weights.weightsIH
	self.weightsHO = weights.weightsHO
	self.biasH = weights.biasH
	self.biasO = weights.biasO
end

-- ========================
-- // MAIN
-- ========================

-- New(): Creates a new neural network.
-- @param InputNodes number Number of input nodes
-- @param HiddenNodes number Number of hidden nodes
-- @param OutputNodes number Number of output nodes
-- @param Seed? number Optional random seed
-- @param PretrainedWeights? table Optional pre-trained weights
-- @return LuaNetwork
function LuaNetwork.New(InputNodes: number, HiddenNodes: number, OutputNodes: number, Seed: number?, PretrainedWeights: {number}?)
	if InputNodes <= 0 or HiddenNodes <= 0 or OutputNodes <= 0 then
		error("InputNodes, HiddenNodes, and OutputNodes must be positive")
	end

	local Random = Seed and Random.new(Seed) or Random.new()

	local self = setmetatable({
		inputNodes = InputNodes,
		hiddenNodes = HiddenNodes,
		outputNodes = OutputNodes,
		weightsIH = {}, -- // Input -> Hidden weights
		weightsHO = {}, -- // Hidden -> Output weights
		biasH = {}, -- // Hidden biases
		biasO = {} -- // Output biases
	}, LuaNetwork)

	if PretrainedWeights then -- From Python
		self:LoadWeights(PretrainedWeights)
	else
		for i = 1, InputNodes do
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

-- Predict(): Feeds input through the network and returns predictions.
-- @param Data: Input values
-- @return outputs: Predicted outputs
function LuaNetwork:Predict(Data: {number}): {number}
	if #Data ~= self.inputNodes then
		error(string.format("Predict: Data length is %d, expected %d input nodes", #Data, self.inputNodes))
	end
	
	local _, outputs = self:ForwardPropagation(Data)
	
	return outputs
end

-- Train(): Runs forward + backward propagation to train the network.
-- @param Data: Input values
-- @param Target: Expected values
-- @param LearningRate: Learning rate (clamped [0.0001, 1])
-- @return cost: The cost value
-- @return outputs: Predicted outputs
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
