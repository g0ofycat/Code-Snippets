--=========================
-- // SERVICES
--=========================

local UserInputService = game:GetService("UserInputService")

--=========================
-- // TYPES
--=========================

local Types = require(script.Types)

--=========================
-- // MAIN MODULE
--=========================

local DrawingPad = {}
DrawingPad.__index = DrawingPad

--=========================
-- // CONSTANTS
--=========================

local DEFAULT_CONFIG: Types.Config = {
	GridSize = 28,
	PredictionCooldown = 0.1,
	NormalizeStats = {
		Mean = 0.1307,
		StdDev = 0.3081
	}
}

--=========================
-- // CONSTRUCTOR
--=========================

-- New(): Creates a new DrawingPad instance
-- @param drawingPadFrame GuiObject: Frame that represents the drawing area
-- @param clearButton GuiButton: Button to clear the drawing pad
-- @param dataLabel TextLabel: Label to display prediction results
-- @param network any: Neural network used for predictions
-- @param config Types.Config?: Optional configuration
-- @return DrawingPad: New instance
function DrawingPad.New(drawingPadFrame: GuiObject, clearButton: GuiButton, dataLabel: TextLabel, network: any, config: Types.Config?): DrawingPad
	local self = setmetatable({} :: any, DrawingPad) :: Types.DrawingPad

	assert(drawingPadFrame, "DrawingPad frame is required")
	assert(clearButton, "Clear button is required")
	assert(dataLabel, "Data label is required")
	assert(network, "Neural network is required")

	config = config or {}
	
	self.GridSize = config.GridSize or DEFAULT_CONFIG.GridSize :: number
	self.PredictionCooldown = config.PredictionCooldown or DEFAULT_CONFIG.PredictionCooldown :: number
	self.NormalizeStats = config.NormalizeStats or DEFAULT_CONFIG.NormalizeStats :: {Mean: number, StdDev: number}

	self.DrawingPadFrame = drawingPadFrame
	self.ClearButton = clearButton
	self.DataLabel = dataLabel
	self.Network = network

	self.Grid = {}
	self.PixelFrames = {}
	self.IsDrawing = false
	self.LastPredictionTime = 0
	self.LastDrawPos = nil

	self:_Initialize()

	return self
end

--=========================
-- // PRIVATE METHODS
--=========================

-- _Initialize(): Initializes the drawing pad (pixel sizes, events, label)
function DrawingPad:_Initialize(): ()
	self:_UpdatePixelSizes()

	self:_ConnectEvents()

	self.DataLabel.Text = "Draw a digit to see predictions"
end

-- _CalculatePixelSize(): Calculates pixel width and height based on pad size and grid size
-- @return {X: number, Y: number}: Pixel dimensions
function DrawingPad:_CalculatePixelSize(): {X: number, Y: number}
	local padSize: Vector2 = self.DrawingPadFrame.AbsoluteSize
	return {
		X = padSize.X / self.GridSize,
		Y = padSize.Y / self.GridSize
	}
end

-- _UpdatePixelSizes(): Updates all pixel frames and the grid
function DrawingPad:_UpdatePixelSizes(): ()
	local pixelSize: {X: number, Y: number} = self:_CalculatePixelSize()

	for _, pixel: Frame in self.PixelFrames do
		pixel:Destroy()
	end
	
	self.PixelFrames = {}

	for y: number = 1, self.GridSize do
		self.Grid[y] = self.Grid[y] or {}
		for x: number = 1, self.GridSize do
			self.Grid[y][x] = self.Grid[y][x] or 0

			local pixel: Frame = Instance.new("Frame")
			pixel.Size = UDim2.new(0, pixelSize.X, 0, pixelSize.Y)
			pixel.Position = UDim2.new(0, (x-1)*pixelSize.X, 0, (y-1)*pixelSize.Y)
			pixel.BackgroundColor3 = Color3.new(1, 1, 1)
			pixel.BackgroundTransparency = 1
			pixel.BorderSizePixel = 1
			pixel.BorderColor3 = Color3.new(0.2, 0.2, 0.2)
			pixel.Name = string.format("Pixel_%d_%d", x, y)
			pixel.Parent = self.DrawingPadFrame

			self.PixelFrames[y .. "_" .. x] = pixel
		end
	end

	self.DataLabel.Text = "Draw a digit to see predictions"
end

-- _ConnectEvents(): Connects input and button events for drawing and clearing
function DrawingPad:_ConnectEvents(): ()
	self.DrawingPadFrame:GetPropertyChangedSignal("AbsoluteSize"):Connect(function()
		self:_UpdatePixelSizes()
	end)

	self.DrawingPadFrame.InputBegan:Connect(function(input: InputObject)
		self:_HandleInput(input, true)
	end)

	self.DrawingPadFrame.InputChanged:Connect(function(input: InputObject)
		self:_HandleInput(input, false)
	end)

	UserInputService.InputEnded:Connect(function(input: InputObject)
		if input.UserInputType == Enum.UserInputType.MouseButton1 or
			input.UserInputType == Enum.UserInputType.Touch then
			if self.IsDrawing then
				self.IsDrawing = false
				self.LastDrawPos = nil
				self:_MakePrediction()
			end
		end
	end)

	self.ClearButton.Activated:Connect(function()
		self:Clear()
	end)
end

-- _ApplyBrush(): Applies brush intensity to a pixel and its neighbors
-- @param x number: X coordinate in the grid
-- @param y number: Y coordinate in the grid
function DrawingPad:_ApplyBrush(x: number, y: number): ()
	if x < 1 or x > self.GridSize or y < 1 or y > self.GridSize then return end

	local function updateSinglePixel(px: number, py: number, intensity: number): ()
		if px >= 1 and px <= self.GridSize and py >= 1 and py <= self.GridSize then
			self.Grid[py][px] = math.min(1, (self.Grid[py][px] or 0) + intensity)
			local pixel: Frame? = self.PixelFrames[py .. "_" .. px]
			if pixel then
				pixel.BackgroundTransparency = 1 - self.Grid[py][px]
			end
		end
	end

	-- // Exact pixel
	
	updateSinglePixel(x, y, 1)
	
	-- // Neighbor pixels
	
	updateSinglePixel(x + 1, y, 0.9)
	updateSinglePixel(x - 1, y, 0.9)
	updateSinglePixel(x, y + 1, 0.9)
	updateSinglePixel(x, y - 1, 0.9)
	
	-- // Diagonal pixels
	
	updateSinglePixel(x + 1, y + 1, 0.5)
	updateSinglePixel(x - 1, y - 1, 0.5)
	updateSinglePixel(x + 1, y - 1, 0.5)
	updateSinglePixel(x - 1, y + 1, 0.5)
end

-- _CenterGridData(): Centers drawn data in the grid based on weighted center of mass
-- @param currentGrid {{number}}: Grid to center
-- @return {{number}}: Centered grid
function DrawingPad:_CenterGridData(currentGrid: {{number}}): {{number}}
	local weightedXSum: number = 0
	local weightedYSum: number = 0
	local totalMass: number = 0
	local tempGrid: {{number}} = {}

	for y: number = 1, self.GridSize do
		tempGrid[y] = {}
		for x: number = 1, self.GridSize do
			local intensity: number = currentGrid[y] and currentGrid[y][x] or 0
			if intensity > 0.01 then
				weightedXSum = weightedXSum + x * intensity
				weightedYSum = weightedYSum + y * intensity
				totalMass += intensity
			end
			tempGrid[y][x] = 0
		end
	end

	if totalMass == 0 then return currentGrid end

	local comX: number = weightedXSum / totalMass
	local comY: number = weightedYSum / totalMass

	local deltaX: number = math.floor(self.GridSize / 2 + 0.5 - comX)
	local deltaY: number = math.floor(self.GridSize / 2 + 0.5 - comY)

	for y: number = 1, self.GridSize do
		for x: number = 1, self.GridSize do
			local oldX: number = x - deltaX
			local oldY: number = y - deltaY

			if oldX >= 1 and oldX <= self.GridSize and oldY >= 1 and oldY <= self.GridSize then
				tempGrid[y][x] = currentGrid[oldY] and currentGrid[oldY][oldX] or 0
			end
		end
	end

	return tempGrid
end

-- _GetGridData(): Flattens the grid into a 1D array for network input
-- @param sourceGrid {{number}}?: Optional grid to use, defaults to current grid
-- @return {number}: Flattened grid data
function DrawingPad:_GetGridData(sourceGrid: {{number}}?): {number}
	local gridToUse: {{number}} = sourceGrid or self.Grid
	local flatData: {number} = {}
	for y: number = 1, self.GridSize do
		for x: number = 1, self.GridSize do
			table.insert(flatData, gridToUse[y] and gridToUse[y][x] or 0)
		end
	end
	return flatData
end

-- _MakePrediction(): Runs the neural network on the current grid and updates DataLabel
function DrawingPad:_MakePrediction(): ()
	local currentTime: number = os.clock()
	if currentTime - self.LastPredictionTime < self.PredictionCooldown then
		return
	end
	self.LastPredictionTime = currentTime

	local success: boolean, result: string = pcall(function(): string
		local centeredGrid: {{number}} = self:_CenterGridData(self.Grid)
		local centeredFlatData: {number} = self:_GetGridData(centeredGrid)

		local maxIntensity: number = 0
		for i: number = 1, #centeredFlatData do
			maxIntensity = math.max(maxIntensity, centeredFlatData[i])
		end

		if maxIntensity > 0.1 then
			for i: number = 1, #centeredFlatData do
				centeredFlatData[i] = centeredFlatData[i] / maxIntensity
			end
		end

		for i: number = 1, #centeredFlatData do
			centeredFlatData[i] = (centeredFlatData[i] - self.NormalizeStats.Mean) / self.NormalizeStats.StdDev
		end

		local outputs: {number} = self.Network:Predict(centeredFlatData)

		local predictedDigit: number = 0
		local maxOutput: number = -math.huge

		if outputs and #outputs > 0 then
			maxOutput = outputs[1]
			predictedDigit = 0
			for i: number = 2, #outputs do
				if outputs[i] >= maxOutput then
					maxOutput = outputs[i]
					predictedDigit = i - 1
				end
			end
		end

		local sortedOutputs: {{index: number, value: number}} = {}
		if outputs then
			for i: number = 1, #outputs do
				table.insert(sortedOutputs, {index = i - 1, value = outputs[i]})
			end
			table.sort(sortedOutputs, function(a: {index: number, value: number}, b: {index: number, value: number}): boolean 
				return a.value > b.value 
			end)
		end

		local confidence: number = (maxOutput > -math.huge) and (maxOutput * 100) or 0
		local displayText: string = string.format("Predicted: %d\nConfidence: %.2f%%\n\nProbabilities:\n",
			predictedDigit, confidence)

		for _, output: {index: number, value: number} in sortedOutputs do
			displayText ..= string.format("%d: %.2f%%\n", output.index, output.value * 100)
		end

		return displayText
	end)

	self.DataLabel.Text = success and result or "Error in prediction. Please try again."
end

-- _HandleInput(): Handles drawing input from mouse or touch
-- @param input InputObject: User input
-- @param isStart boolean: True if input just started
function DrawingPad:_HandleInput(input: InputObject, isStart: boolean): ()
	if (input.UserInputType == Enum.UserInputType.MouseButton1 or
		input.UserInputType == Enum.UserInputType.Touch) and isStart then
		self.IsDrawing = true
		self.LastDrawPos = nil
	elseif self.IsDrawing and (input.UserInputType == Enum.UserInputType.MouseMovement or
		input.UserInputType == Enum.UserInputType.Touch) then

		local pos: Vector3 = input.Position
		local padPos: Vector2 = self.DrawingPadFrame.AbsolutePosition
		local relativeX: number = pos.X - padPos.X
		local relativeY: number = pos.Y - padPos.Y
		local pixelSize: {X: number, Y: number} = self:_CalculatePixelSize()
		local gridX: number = math.floor(relativeX / pixelSize.X) + 1
		local gridY: number = math.floor(relativeY / pixelSize.Y) + 1

		local currentDrawPos: Vector2 = Vector2.new(gridX, gridY)

		if self.LastDrawPos then
			local delta: Vector2 = currentDrawPos - self.LastDrawPos
			local steps: number = math.max(math.abs(delta.X), math.abs(delta.Y))
			if steps > 0 then
				for i: number = 0, steps do
					local t: number = i / steps
					local interpX: number = math.floor(self.LastDrawPos.X + delta.X * t + 0.5)
					local interpY: number = math.floor(self.LastDrawPos.Y + delta.Y * t + 0.5)
					self:_ApplyBrush(interpX, interpY)
				end
			else
				self:_ApplyBrush(gridX, gridY)
			end
		else
			self:_ApplyBrush(gridX, gridY)
		end
		self.LastDrawPos = currentDrawPos

		self:_MakePrediction()
	end
end

--=========================
-- // PUBLIC METHODS
--=========================

-- Clear(): Clears the drawing pad and resets the grid
function DrawingPad:Clear(): ()
	for y: number = 1, self.GridSize do
		if self.Grid[y] then
			for x: number = 1, self.GridSize do
				self.Grid[y][x] = 0
				local pixel: Frame? = self.PixelFrames[y .. "_" .. x]
				if pixel then
					pixel.BackgroundTransparency = 1
				end
			end
		end
	end

	self.LastDrawPos = nil
	self.DataLabel.Text = "Draw a digit to see predictions"
end

-- GetGridData(): Returns flattened grid data for external use
-- @return {number}: Flattened grid data
function DrawingPad:GetGridData(): {number}
	return self:_GetGridData()
end

-- SetNormalizationStats(): Updates normalization mean and standard deviation
-- @param mean number: New mean
-- @param stdDev number: New standard deviation
function DrawingPad:SetNormalizationStats(mean: number, stdDev: number): ()
	self.NormalizeStats = {
		Mean = mean,
		StdDev = stdDev
	}
end

-- Destroy(): Removes all pixel frames and clears the grid
function DrawingPad:Destroy(): ()
	for _, pixel: Frame in self.PixelFrames do
		pixel:Destroy()
	end

	self.PixelFrames = {}
	self.Grid = {}
end

return DrawingPad
