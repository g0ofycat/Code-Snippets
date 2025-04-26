--[[ TODO:

	- Multiple Hidden layers	

]]

--[[ NOTES:
	
	- Each neuron is connected to every single one in the next layer and so on
	- Weights: The strength of the 	connections between each neuron
	- Biases: Makes the result more offset, allowing for (possibly) better responses
	- Neurons: A single neuron that makes up a whole neural network, each neural is fed a weighted sum from the previous layer, it's then added plus the bias. (WeightedSum + Bias)
	- Sigmoid: An activation for the neuron, it returns a value from 0-1. It allows networks non-linearity (Learning Complex Patterns)
	- Cost: Costs are useful since it provides the error margin through a number using the Output of a neural network and comparing it to the actual output given. It's used to refine the output and eventully get the right Output
	
	Currently using: Xavier Initialization, Sigmoids, Softmax, Cross-Entropy to calculate cost, and [-0.1, 0.1] for biases.
]]

local LuaNetwork = {}

LuaNetwork.__index = LuaNetwork

---------------------------------------------
-- // Helper Functions
---------------------------------------------

function Sigmoid(x: number): number -- // Activation function in each neuron, both equations produce the same output although both equations are better based on if they're positive or negative
	if x >= 0 then
		return 1 / (1 + math.exp(-x))
	else
		return math.exp(x) / (1 + math.exp(x))
	end
end

function SigmoidDerivative(y: number): number -- // Used for backpropagation to adjust weights and bias
	return y * (1 - y)
end

function Softmax(inputs: {number}): {number}
	local max = inputs[1]
	for i = 2, #inputs do
		if inputs[i] > max then max = inputs[i] end
	end
	local exp_sum = 0
	local outputs = {}
	for i = 1, #inputs do
		outputs[i] = math.exp(inputs[i] - max)
		exp_sum += outputs[i]
	end
	for i = 1, #inputs do
		outputs[i] /= exp_sum
	end
	return outputs
end

function CostCalculator(Outputs: {number}, Expected: {number}): number -- // The (average) amount of error between what my module predicts VS the actual value. (Uses Cross-Entropy)
	local sum = 0
	for i = 1, #Outputs do
		local pred = math.clamp(Outputs[i], 1e-7, 1 - 1e-7)
		sum += -Expected[i] * math.log(pred)
	end
	return sum / #Outputs
end

function LuaNetwork:ForwardPropagation(Data: {number}): ({number}, {number})
	local hiddenInputs = {}
	local hiddenOutputs = {}
	for j = 1, self.hiddenNodes do
		local sum = 0
		for i = 1, self.inputNodes do
			sum += Data[i] * self.weightsIH[i][j]
		end
		hiddenInputs[j] = sum + self.biasH[j]
		hiddenOutputs[j] = Sigmoid(hiddenInputs[j])
	end

	local outputInputs = {}
	local outputs = {}
	for k = 1, self.outputNodes do
		local sum = 0
		for j = 1, self.hiddenNodes do
			sum += hiddenOutputs[j] * self.weightsHO[j][k]
		end
		outputInputs[k] = sum + self.biasO[k]
	end
	outputs = Softmax(outputInputs)

	return hiddenOutputs, outputs
end

function LuaNetwork:BackPropagation(Data: {number}, Target: {number}, HiddenOutputs: {number}, Outputs: {number}, LearningRate: number)
	-- // Backpropagation: Output layer
	-- // For Softmax with Cross-Entropy, the gradient simplifies to (output - target)
	local outputGradients = {}
	for k = 1, self.outputNodes do
		outputGradients[k] = Outputs[k] - Target[k]

		for j = 1, self.hiddenNodes do
			self.weightsHO[j][k] -= LearningRate * outputGradients[k] * HiddenOutputs[j]
		end
		self.biasO[k] -= LearningRate * outputGradients[k]
	end

	-- // Backpropagation: Hidden layer
	local hiddenGradients = {}
	for j = 1, self.hiddenNodes do
		local error = 0
		for k = 1, self.outputNodes do
			error += outputGradients[k] * self.weightsHO[j][k]
		end
		hiddenGradients[j] = error * SigmoidDerivative(HiddenOutputs[j])
		for i = 1, self.inputNodes do
			self.weightsIH[i][j] -= LearningRate * hiddenGradients[j] * Data[i]
		end
		self.biasH[j] -= LearningRate * hiddenGradients[j]
	end
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
function LuaNetwork.New(InputNodes: number, HiddenNodes: number, OutputNodes: number, Seed: number) -- // Constructor
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

	-- // Init Weights and Biases for all tables (-1, 1), (Weights are initialized by Uniform Xavier)

	for i = 1, InputNodes do -- // Weights for Hidden Neuron Connections
		self.weightsIH[i] = {}
		for j = 1, HiddenNodes do
			local scaleIH = math.sqrt(6 / (InputNodes + HiddenNodes))
			self.weightsIH[i][j] = Random:NextNumber(-scaleIH, scaleIH)
		end
	end

	for j = 1, HiddenNodes do -- // Weights for the hidden nodes to the output
		self.weightsHO[j] = {}
		for k = 1, OutputNodes do
			local scaleHO = math.sqrt(6 / (HiddenNodes + OutputNodes))
			self.weightsHO[j][k] = Random:NextNumber(-scaleHO, scaleHO)
		end
	end

	for j = 1, HiddenNodes do -- // Bias for hiddens
		self.biasH[j] = Random:NextNumber(-0.1, 0.1)
	end

	for k = 1, OutputNodes do -- // Bias for outputs
		self.biasO[k] = Random:NextNumber(-0.1, 0.1)
	end

	return self
end

---------------------------------------------
-- // Propagations
---------------------------------------------

--[[
	:Predict() Used to feed in data to the Neural Network for actual training
	
	- Data: A table of numbers for the neural network to learn from
]]
function LuaNetwork:Predict(Data: {number}): {number} -- // Forward Propagation, I to H
	if #Data ~= self.inputNodes then
		error("Data length must equal InputNodes")
	end

	local _, outputs = self:ForwardPropagation(Data)
	return outputs
end

--[[
	:Train() Used to train the data using Forward and Backpropagation
	
	- Data: A table of numbers for the neural network to learn from
	- Target: A table that has the desired values you want the Neural Network to learn
	- LearningRate: How quickly the Neural Network adjusts it's parameters to learn
]]
function LuaNetwork:Train(Data: {number}, Target: {number}, LearningRate: number): (number, {number})
	if #Data ~= self.inputNodes then
		error("Data length must equal InputNodes")
	end
	if #Target ~= self.outputNodes then
		error("Target length must equal OutputNodes")
	end
	if LearningRate < 0.0001 or LearningRate > 1 then
		warn("LearningRate should typically be between 0.0001 and 1 for stability")
		LearningRate = math.clamp(LearningRate, 0.0001, 1)
	end

	-- // Forward propagation
	local hiddenOutputs, outputs = self:ForwardPropagation(Data)

	local cost = CostCalculator(outputs, Target)

	-- // Backpropagation
	self:BackPropagation(Data, Target, hiddenOutputs, outputs, LearningRate)

	return cost, outputs
end

return LuaNetwork
