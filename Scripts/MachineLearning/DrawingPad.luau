--!native
--!optimize 2

local DrawingPad = script.Parent
local ClearButton = DrawingPad.Parent.Clear
local Percentages = DrawingPad.Parent.Data
local UserInputService = game:GetService("UserInputService")
local LuaNetwork = require(game.ReplicatedStorage.LuaNetwork)
local Weights = require(game.ReplicatedStorage.Weights)

local network = LuaNetwork.New(784, 256, 10, nil, Weights) -- // 784 Input (28x28), 256 Hidden Nodes (Update with new biases if you wanna change this), 10 Output Nodes. (0-9). nil Parameter is the Seed for Random.new() if you wan't to train it without pre determined weights and biases.

local GRID_SIZE = 28 -- // Size of grid (28x28)
local grid = {}
local pixelFrames = {} -- // A table storing each pixel frame
local isDrawing = false
local lastPredictionTime = 0
local PREDICTION_COOLDOWN = 0.1 -- // Intervals for the predictions to update
local lastDrawPos = nil

local function calculatePixelSize() -- // Returns the drawing pad size (in pixels) divided by the grid size. Maps screen coordinates to actual grid
	local padSize = DrawingPad.AbsoluteSize
	return {
		X = padSize.X / GRID_SIZE,
		Y = padSize.Y / GRID_SIZE
	}
end

local function updatePixelSizes() -- // If the absolute size is messed with then we need to reset all pixels. This is because a different sized grid can mess up the output. Initalizes every pixel frame
	local pixelSize = calculatePixelSize()
	for _, pixel in pairs(pixelFrames) do
		pixel:Destroy()
	end
	pixelFrames = {}
	for y = 1, GRID_SIZE do
		grid[y] = grid[y] or {}
		for x = 1, GRID_SIZE do
			grid[y][x] = grid[y][x] or 0
			local pixel = Instance.new("Frame")
			pixel.Size = UDim2.new(0, pixelSize.X, 0, pixelSize.Y)
			pixel.Position = UDim2.new(0, (x-1)*pixelSize.X, 0, (y-1)*pixelSize.Y)
			pixel.BackgroundColor3 = Color3.new(1, 1, 1)
			pixel.BackgroundTransparency = 1
			pixel.BorderSizePixel = 1
			pixel.BorderColor3 = Color3.new(0.2, 0.2, 0.2)
			pixel.Name = string.format("Pixel_%d_%d", x, y)
			pixel.Parent = DrawingPad
			pixelFrames[y .. "_" .. x] = pixel
		end
	end
	Percentages.Text = "Draw a digit to see predictions"
end

updatePixelSizes()

DrawingPad:GetPropertyChangedSignal("AbsoluteSize"):Connect(updatePixelSizes)

local function applyBrush(x, y) -- // Drawing Functionality, kinda like a soft-brush effect
	if x < 1 or x > GRID_SIZE or y < 1 or y > GRID_SIZE then return end

	local function updateSinglePixel(px, py, intensity)
		if px >= 1 and px <= GRID_SIZE and py >= 1 and py <= GRID_SIZE then -- // Update the certain pixels transparency based on intensity and its X and Y
			grid[py][px] = math.min(1, (grid[py][px] or 0) + intensity)
			local pixel = pixelFrames[py .. "_" .. px]
			if pixel then
				pixel.BackgroundTransparency = 1 - grid[py][px]
			end
		end
	end
	 
	updateSinglePixel(x, y, 1) -- // Exact pixel ur on
	-- // Neighbor Pixels
	updateSinglePixel(x + 1, y, 0.9)
	updateSinglePixel(x - 1, y, 0.9)
	updateSinglePixel(x, y + 1, 0.9)
	updateSinglePixel(x, y - 1, 0.9)
	-- // Diagonal Pixels
	updateSinglePixel(x + 1, y + 1, 0.5)
	updateSinglePixel(x - 1, y - 1, 0.5)
	updateSinglePixel(x + 1, y - 1, 0.5)
	updateSinglePixel(x - 1, y + 1, 0.5)
end

local function centerGridData(currentGrid) -- // Useful since offset centered drawn numbers don't matter now. The MNIST data set is centered although I did add some data augmentation to the MNISTData and some noise
	local weightedXSum = 0
	local weightedYSum = 0
	local totalMass = 0
	local tempGrid = {}
	
	-- // Gets the weighted intensity of all pixels based on the X or Y and the intensity
	for y = 1, GRID_SIZE do
		tempGrid[y] = {}
		for x = 1, GRID_SIZE do
			local intensity = currentGrid[y] and currentGrid[y][x] or 0
			if intensity > 0.01 then
				weightedXSum = weightedXSum + x * intensity
				weightedYSum = weightedYSum + y * intensity
				totalMass += intensity
			end
			tempGrid[y][x] = 0
		end
	end

	if totalMass == 0 then return currentGrid end -- // No data
	
	local comX = weightedXSum / totalMass
	local comY = weightedYSum / totalMass

	local deltaX = math.floor(GRID_SIZE / 2 + 0.5 - comX)
	local deltaY = math.floor(GRID_SIZE / 2 + 0.5 - comY)
	
	for y = 1, GRID_SIZE do
		for x = 1, GRID_SIZE do
			local oldX = x - deltaX
			local oldY = y - deltaY

			if oldX >= 1 and oldX <= GRID_SIZE and oldY >= 1 and oldY <= GRID_SIZE then -- // No out of bounds
				tempGrid[y][x] = currentGrid[oldY] and currentGrid[oldY][oldX] or 0 -- // Set the coords of the new centered data
			end
		end
	end
	
	return tempGrid -- // Returns a new grid in which all data is in the center
end

local function getGridData(sourceGrid) -- // Returns the flattened version of the pixel intensity grid (0-1), similar to the data structure of the MNISTData module
	local gridToUse = sourceGrid or grid
	local flatData = {}
	for y = 1, GRID_SIZE do
		for x = 1, GRID_SIZE do
			table.insert(flatData, gridToUse[y] and gridToUse[y][x] or 0)
		end
	end
	return flatData
end


local function makePrediction() -- // Uses the NN to predict what number it is by normalizing it and centering data AND then feeds it into the NN
	local currentTime = os.clock()
	if currentTime - lastPredictionTime < PREDICTION_COOLDOWN then
		return
	end
	lastPredictionTime = currentTime

	local success, result = pcall(function()
		local centeredGrid = centerGridData(grid) -- // Center grid data for better accuracies
		local centeredFlatData = getGridData(centeredGrid) -- // Flatten in 1D Array
		
		local maxIntensity = 0
		
		for i = 1, #centeredFlatData do -- // Find max intensity to normalize each pixel value. It allows for the intensities to be consistent which make better results
			maxIntensity = math.max(maxIntensity, centeredFlatData[i])
		end

		if maxIntensity > 0.1 then -- // Normalize (0-1). Must be over 0.1 which makes it process meaningful data
			for i = 1, #centeredFlatData do
				centeredFlatData[i] = centeredFlatData[i] / maxIntensity
			end
		end

		for i = 1, #centeredFlatData do
			centeredFlatData[i] = (centeredFlatData[i] - 0.1307) / 0.3081 -- // Mean 0.1307 and standard deviation of 0.3081. What this does it that it basically aligns the input data with the MNIST data set which is EXTREMLEY important since it allows for accurate predictions.
		end

		local outputs = network:Predict(centeredFlatData) -- // Now predicts based on the data. Returns a table of 10 values representing the score of each number

		local predictedDigit = 0
		local maxOutput = -math.huge
		
		-- // See if it's a valid value and find the digit with the highest	score which is the predicted digit
		if outputs and #outputs > 0 then
			maxOutput = outputs[1]
			predictedDigit = 0
			for i = 2, #outputs do
				if outputs[i] >= maxOutput then
					maxOutput = outputs[i]
					predictedDigit = i - 1
				end
			end
		end
		
		-- // Sort outputs based on probabilities
		local sortedOutputs = {}
		
		if outputs then
			for i = 1, #outputs do
				table.insert(sortedOutputs, {index = i - 1, value = outputs[i]})
			end
			table.sort(sortedOutputs, function(a, b) return a.value > b.value end)
		end

		local confidence = (maxOutput > -math.huge) and (maxOutput * 100) or 0
		local displayText = string.format("Predicted: %d\nConfidence: %.2f%%\n\nProbabilities:\n",
			predictedDigit, confidence)

		for _, output in ipairs(sortedOutputs) do
			displayText ..= string.format("%d: %.2f%%\n", output.index, output.value * 100)
		end

		return displayText
	end)

	Percentages.Text = success and result or "Error in prediction. Please try again."
end

local function handleInput(input, isStart) -- // Handle both touch and mouse inputs
	if (input.UserInputType == Enum.UserInputType.MouseButton1 or
		input.UserInputType == Enum.UserInputType.Touch) and isStart then
		isDrawing = true
		lastDrawPos = nil
	elseif isDrawing and (input.UserInputType == Enum.UserInputType.MouseMovement or
		input.UserInputType == Enum.UserInputType.Touch) then
		local pos = input.Position
		local padPos = DrawingPad.AbsolutePosition
		local relativeX = pos.X - padPos.X
		local relativeY = pos.Y - padPos.Y
		local pixelSize = calculatePixelSize()
		local gridX = math.floor(relativeX / pixelSize.X) + 1
		local gridY = math.floor(relativeY / pixelSize.Y) + 1

		local currentDrawPos = Vector2.new(gridX, gridY)

		if lastDrawPos then -- // I've expereinced a bug where the drawing sometimes just stops / cuts off, messing your number up. I fixed this by getting the amount of steps and filling in missing spots, making smooth lines
			local delta = currentDrawPos - lastDrawPos
			local steps = math.max(math.abs(delta.X), math.abs(delta.Y))
			if steps > 0 then
				for i = 0, steps do
					local t = i / steps
					local interpX = math.floor(lastDrawPos.X + delta.X * t + 0.5)
					local interpY = math.floor(lastDrawPos.Y + delta.Y * t + 0.5)
					applyBrush(interpX, interpY) -- // The missing brush stroke
				end
			else
				applyBrush(gridX, gridY)
			end
		else
			applyBrush(gridX, gridY)
		end
		lastDrawPos = currentDrawPos

		makePrediction()
	end
end

-- // Inputs

DrawingPad.InputBegan:Connect(function(input) handleInput(input, true) end)
DrawingPad.InputChanged:Connect(function(input) handleInput(input, false) end)

UserInputService.InputEnded:Connect(function(input)
	if input.UserInputType == Enum.UserInputType.MouseButton1 or
		input.UserInputType == Enum.UserInputType.Touch then
		if isDrawing then
			isDrawing = false
			lastDrawPos = nil
			makePrediction()
		end
	end
end)

ClearButton.Activated:Connect(function() -- // Clearing functionality
	for y = 1, GRID_SIZE do
		if grid[y] then
			for x = 1, GRID_SIZE do
				grid[y][x] = 0
				local pixel = pixelFrames[y .. "_" .. x]
				if pixel then
					pixel.BackgroundTransparency = 1
				end
			end
		end
	end
	lastDrawPos = nil
	Percentages.Text = "Draw a digit to see predictions"
end)
