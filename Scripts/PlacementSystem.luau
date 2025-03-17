-- // Variables

local CollectionService = game:GetService("CollectionService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local TweenService = game:GetService("TweenService")
local UserInputService = game:GetService("UserInputService")
local camera = workspace.CurrentCamera

local player = Players.LocalPlayer
local pickedUpModel = nil
local canRotate = true
local db = false
local Anchored = false

-- // Script Presets

local targetOffset = CFrame.new(0, 0, -5)
local RotateIncrement = 45
local MaxDistance = CFrame.new(0, 0, -15)
local MinDistance = CFrame.new(0, 0, -3)
local LerpSpeed = 0.05
local maxGrabDistance = 20

-- // Keybinds for Rotating Objects

local Keybinds = {
	X = Enum.KeyCode.X,
	Y = Enum.KeyCode.Y,
	Z = Enum.KeyCode.Z,
	E = Enum.KeyCode.E,
	R = Enum.KeyCode.R,
	M = Enum.KeyCode.M
}

local highlightTweenInfo = TweenInfo.new(0.5, Enum.EasingStyle.Quad, Enum.EasingDirection.Out, 0, false, 0)

-- // Helper Functions

local function createTween(object, properties)
	return TweenService:Create(object, highlightTweenInfo, properties)
end

local function addHighlight(model)
	local highlight = Instance.new("Highlight")
	highlight.Parent = model
	highlight.FillTransparency = 1
	highlight.OutlineTransparency = 1
	highlight.DepthMode = "Occluded"

	createTween(highlight, {FillTransparency = 0.5, OutlineTransparency = 0}):Play()
end

local function removeHighlight(model)
	local highlight = model:FindFirstChild("Highlight")
	if highlight then
		local fillTween = createTween(highlight, {FillTransparency = 1, OutlineTransparency = 1})

		fillTween:Play()

		fillTween.Completed:Connect(function()
			highlight:Destroy()
		end)
	end
end

local function detectCollisionWithShape(model)
	if not model.PrimaryPart then return false end

	local modelCFrame, size = model:GetBoundingBox()
	local overlapParams = OverlapParams.new()
	overlapParams.FilterDescendantsInstances = {model, player.Character}
	overlapParams.FilterType = Enum.RaycastFilterType.Exclude

	local overlappingParts = workspace:GetPartBoundsInBox(modelCFrame, size, overlapParams)

	for _, part in ipairs(overlappingParts) do
		local parent = part.Parent
		if not Players:GetPlayerFromCharacter(parent) and parent ~= player.Character then
			return true
		end
	end
	return false
end

local function pickUp(model)
	if pickedUpModel or db then return end
	local character = player.Character
	if not character or not character:FindFirstChild("HumanoidRootPart") then return end

	local distance = (character.HumanoidRootPart.Position - model.PrimaryPart.Position).Magnitude
	if distance > maxGrabDistance then return end

	db = true
	pickedUpModel = model
	game.ReplicatedStorage:WaitForChild("Events").Grab:FireServer(pickedUpModel)
	script:WaitForChild("Click"):Play()

	addHighlight(model)

	local guiElements = {"GrabControls", "MobileButtons"}
	for _, element in ipairs(guiElements) do
		local canvasGroup = player.PlayerGui:FindFirstChild(element).CanvasGroup
		createTween(canvasGroup, {GroupTransparency = 0}):Play()
	end

	local raycastParams = RaycastParams.new()
	raycastParams.FilterDescendantsInstances = {model, character}
	raycastParams.FilterType = Enum.RaycastFilterType.Exclude

	local collisionMargin = 0.1

	RunService.Heartbeat:Connect(function()
		if not pickedUpModel then return end

		if character and character:FindFirstChild("HumanoidRootPart") and camera then
			local targetPosition = (camera.CFrame * targetOffset).Position
			local currentPosition = model.PrimaryPart.Position
			local direction = targetPosition - currentPosition

			if direction.Magnitude == 0 then return end

			local currentOrientation = model.PrimaryPart.CFrame - model.PrimaryPart.Position
			local normalizedDirection = direction.Unit

			local raycastResult = workspace:Raycast(currentPosition, normalizedDirection * direction.Magnitude, raycastParams)

			if raycastResult then
				local adjustedPosition = currentPosition:Lerp(raycastResult.Position - (normalizedDirection * collisionMargin), LerpSpeed)
				model:SetPrimaryPartCFrame(CFrame.new(adjustedPosition) * currentOrientation)
			else
				local lerpedPosition = currentPosition:Lerp(targetPosition, LerpSpeed)
				model:SetPrimaryPartCFrame(CFrame.new(lerpedPosition) * currentOrientation)
			end

			if detectCollisionWithShape(model) then
				Anchored = true
				createTween(pickedUpModel:FindFirstChild("Highlight"), {FillColor = Color3.fromRGB(26, 255, 0)}):Play()
			else
				Anchored = false
				createTween(pickedUpModel:FindFirstChild("Highlight"), {FillColor = Color3.fromRGB(255, 255, 255)}):Play()
			end
		end
	end)
end

local function drop()
	if pickedUpModel then
		local guiElements = {"GrabControls", "MobileButtons"}
		for _, element in ipairs(guiElements) do
			local canvasGroup = player.PlayerGui:FindFirstChild(element).CanvasGroup
			createTween(canvasGroup, {GroupTransparency = 1}):Play()
		end

		game.ReplicatedStorage:WaitForChild("Events").Dropped:FireServer(pickedUpModel, pickedUpModel.PrimaryPart.CFrame, Anchored)
		script:WaitForChild("Click"):Play()
		removeHighlight(pickedUpModel)

		pickedUpModel = nil
		task.wait(1)
		db = false
	end
end

local function getLookedAtModel()
	local raycastParams = RaycastParams.new()
	raycastParams.FilterDescendantsInstances = {player.Character}
	raycastParams.IgnoreWater = true

	local result = workspace:Raycast(camera.CFrame.Position, camera.CFrame.LookVector * 20, raycastParams)
	if result and result.Instance and CollectionService:HasTag(result.Instance.Parent, "CanGrab") then
		return result.Instance.Parent
	end
	return nil
end

-- // Event Handlers

RunService.Heartbeat:Connect(function()
	local modelToPickUp = getLookedAtModel()
	local canvasGroupGrab = player.PlayerGui:FindFirstChild("MobileButtons").CanvasGroupGrab

	if modelToPickUp and not pickedUpModel then
		canvasGroupGrab.GrabButton.ButtonText.Text = "Grab"
		createTween(canvasGroupGrab, {GroupTransparency = 0}):Play()
	elseif pickedUpModel then
		canvasGroupGrab.GrabButton.ButtonText.Text = "Drop"
	else
		createTween(canvasGroupGrab, {GroupTransparency = 1}):Play()
	end
end)

player.PlayerGui:FindFirstChild("MobileButtons").CanvasGroupGrab.GrabButton.Activated:Connect(function()
	if pickedUpModel then
		drop()
	else
		local model = getLookedAtModel()
		if model then pickUp(model) end
	end
end)

player:GetMouse().Button1Down:Connect(function()
	if UserInputService.TouchEnabled then return end
	if pickedUpModel then
		drop()
	else
		local target = player:GetMouse().Target
		if target and target:IsDescendantOf(workspace) then
			local model = target:FindFirstAncestorOfClass("Model")
			if model and CollectionService:HasTag(model, "CanGrab") then
				pickUp(model)
			end
		end
	end
end)

local function rotatePickedUpModel(rotationAxis, rotationDirection)
	if pickedUpModel and pickedUpModel.PrimaryPart then
		local currentCFrame = pickedUpModel.PrimaryPart.CFrame
		local rotationAmount = math.rad(RotateIncrement)
		script:WaitForChild("Click"):Play()
		local rotation = CFrame.Angles(
			rotationAxis == "X" and rotationAmount * rotationDirection or 0,
			rotationAxis == "Y" and rotationAmount * rotationDirection or 0,
			rotationAxis == "Z" and rotationAmount * rotationDirection or 0
		)

		local targetCFrame = (currentCFrame * rotation)

		local function roundCFrame(cframe, precision)
			local position = cframe.Position
			local roll, pitch, yaw = cframe:toEulerAnglesXYZ()

			local roundedAngles = Vector3.new(
				math.round(roll / precision) * precision,
				math.round(pitch / precision) * precision,
				math.round(yaw / precision) * precision
			)

			return CFrame.new(position) * CFrame.Angles(roundedAngles.X, roundedAngles.Y, roundedAngles.Z)
		end

		targetCFrame = roundCFrame(targetCFrame, math.rad(RotateIncrement))

		local tweenInfo = TweenInfo.new(0.5, Enum.EasingStyle.Quad, Enum.EasingDirection.Out)
		local tweenGoal = { CFrame = targetCFrame}
		local tween = TweenService:Create(pickedUpModel.PrimaryPart, tweenInfo, tweenGoal)

		tween:Play()

		canRotate = true
	end
end

local function resetPickedUpModelOrientation()
	if pickedUpModel and pickedUpModel.PrimaryPart then
		local targetCFrame = CFrame.new(pickedUpModel.PrimaryPart.Position)
		script:WaitForChild("Click"):Play()
		local tweenInfo = TweenInfo.new(0.5, Enum.EasingStyle.Quad, Enum.EasingDirection.Out)
		local tweenGoal = { CFrame = targetCFrame }
		local tween = TweenService:Create(pickedUpModel.PrimaryPart, tweenInfo, tweenGoal)

		tween:Play()
	end
end

UserInputService.InputBegan:Connect(function(input, gp)
	if not pickedUpModel or gp then return end

	if input.KeyCode == Keybinds.E then
		if targetOffset ~= MaxDistance then
			targetOffset = targetOffset * CFrame.new(0, 0, -1)
		end
	elseif input.KeyCode == Keybinds.R then
		if targetOffset ~= MinDistance then
			targetOffset = targetOffset * CFrame.new(0, 0, 1)
		end
	elseif input.KeyCode == Keybinds.M then
		resetPickedUpModelOrientation()
	elseif canRotate then
		if input.KeyCode == Keybinds.X then
			rotatePickedUpModel("X", 1)
		elseif input.KeyCode == Keybinds.Y then
			rotatePickedUpModel("Y", 1)
		elseif input.KeyCode == Keybinds.Z then
			rotatePickedUpModel("Z", 1)
		end
	end
end)

-- // Mobile Button Functionality

local mobileButtons = player.PlayerGui:WaitForChild("MobileButtons").CanvasGroup
mobileButtons.FarButton.Activated:Connect(function()
	if not pickedUpModel then return end
	if targetOffset ~= MaxDistance then
		targetOffset = targetOffset * CFrame.new(0, 0, -1)
	end
	script:WaitForChild("Click"):Play()
end)

mobileButtons.CloserButton.Activated:Connect(function()
	if not pickedUpModel then return end
	if targetOffset ~= MinDistance then
		targetOffset = targetOffset * CFrame.new(0, 0, 1)
	end
	script:WaitForChild("Click"):Play()
end)

mobileButtons.X.Activated:Connect(function()
	if not pickedUpModel or not canRotate then return end
	rotatePickedUpModel("X", 1)
end)

mobileButtons.Y.Activated:Connect(function()
	if not pickedUpModel or not canRotate then return end
	rotatePickedUpModel("Y", 1)
end)

mobileButtons.Z.Activated:Connect(function()
	if not pickedUpModel or not canRotate then return end
	rotatePickedUpModel("Z", 1)
end)

mobileButtons.Reset.Activated:Connect(function()
	if not pickedUpModel or not canRotate then return end
	resetPickedUpModelOrientation()
end)
