-- // Everything in this code could obviously be much more optomized like using GroundControllers, Character Controllers, etc. This purpose of this was to create a whole movement engine from scratch only using Robloxs physics including gravity, movement, etc. WIP.
-- // Services and References
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")
local UserInputService = game:GetService("UserInputService")
local moveRigEvent = ReplicatedStorage.Events:WaitForChild("MoveRigEvent")
local cam = workspace.CurrentCamera
local player = Players.LocalPlayer
local character = player.Character or player.CharacterAdded:Wait()
local humanoid = character:WaitForChild("Humanoid")
local humanoidRootPart = character:WaitForChild("HumanoidRootPart")
local rigHead = workspace:WaitForChild("Rig_" .. player.Name):WaitForChild("Head")
local rigBody = workspace:WaitForChild("Rig_" .. player.Name):WaitForChild("Body")
local PlayerGui = player.PlayerGui
local VectorForces = PlayerGui.Keys:WaitForChild("VectorForces")

-- // Movement Variables
local baseSpeed = 30  -- // Base movement speed
local acceleration = 3  -- // Rate of speed increase
local deceleration = 6  -- // Rate of speed decrease
local turnSpeed = 8  -- // Speed of rotation
local gravityForce = workspace.Gravity  -- // Gravity strength from workspace
local groundCheckDistance = 2.25  -- // Distance to check for ground
local terminalVelocity = -700  -- // Maximum fall speed
local airResistance = 80  -- // Resistance when in air
local jumpPower = 50  -- // Force of jump
local AIR_CONTROL = 14  -- // Control factor while in air
local SLIDE_FRICTION = 0.1  -- // Low friction for sliding
local SLIDE_THRESHOLD = 60  -- // Degrees; slope angle to trigger sliding

-- // Network optimization
local sendInterval = 1/30  -- // How fast the client updates the server, 30 FPS
local movementThreshold = 0.01  -- // Increased threshold for more efficient networking
local lastSentTime = 0
local lastSentVelocity = Vector3.zero
local lastSentCFrame = CFrame.new()

-- // Performance measures
local groundCheckInterval = 1/10  -- // Check ground at 30 FPS, optomization reasons
local lastGroundCheckTime = 0
local cachedGroundInfo = {false, false, 0}  -- // {isGrounded, canStand, slopeAngle}

-- // Checks
local currentVelocity = Vector3.zero  -- // Current movement velocity
local currentVerticalVelocity = 0  -- // Vertical component of velocity
local lastCFrame = CFrame.new()  -- // Last frame's CFrame for interpolation
local isGrounded = false  -- // Whether character is on ground
local canJump = true  -- // Whether character can jump
local isSliding = false  -- // Whether character is sliding
local raycastParams = RaycastParams.new()  -- // Parameters for raycasting
raycastParams.FilterType = Enum.RaycastFilterType.Exclude  -- // Exclude filter for raycast
raycastParams.FilterDescendantsInstances = {rigHead, rigBody, character}

-- // Memory management
local connections = {}  -- // Store connections for cleanup

local function checkGround()
	-- // Raycasting every RenderStepped takes up alot of memory, so we can apply it at a fixed rate. (30 FPS)
	local currentTime = tick()
	if currentTime - lastGroundCheckTime >= groundCheckInterval then
		local raycastResult = workspace:Raycast(rigBody.Position, Vector3.new(0, -groundCheckDistance, 0), raycastParams)
		if raycastResult then
			local normal = raycastResult.Normal
			local slopeAngle = math.deg(math.acos(normal.Y))  -- // Angle from vertical
			cachedGroundInfo = {true, slopeAngle <= 90, slopeAngle}  -- // isGrounded, canStand, slopeAngle
		else
			cachedGroundInfo = {false, false, 0}
		end
		lastGroundCheckTime = currentTime
	end
	return unpack(cachedGroundInfo)
end

-- // Cleanup function
local function cleanupConnections()
	for _, connection in pairs(connections) do
		if connection.Connected then
			connection.Disconnect(connection)
		end
	end
	connections = {}
end

-- // Main physics update function with all the improvements
local function updatePhysics(dt)
	local moveDir = humanoid.MoveDirection
	local moveMagnitude = moveDir.Magnitude

	-- // Normal detailed physics
	local grounded, canStand, slopeAngle = checkGround()
	local justLanded = grounded and not isGrounded
	isGrounded = grounded
	isSliding = grounded and not canStand and slopeAngle > SLIDE_THRESHOLD
	local horizontalVelocity = Vector3.new(currentVelocity.X, 0, currentVelocity.Z)

	if grounded then
		if justLanded or currentVerticalVelocity < 0 then
			currentVerticalVelocity = 0
			canJump = true
		end

		if isSliding then
			local slopeDir = Vector3.new(horizontalVelocity.X, 0, horizontalVelocity.Z).Unit
			if slopeDir.Magnitude == 0 then slopeDir = -lastCFrame.LookVector end
			horizontalVelocity += slopeDir * (gravityForce * math.sin(math.rad(slopeAngle)) * dt)
			horizontalVelocity *= (1 - SLIDE_FRICTION * dt)
		elseif moveMagnitude > 0.1 then
			local worldMoveDir = lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit
			local targetVelocity = worldMoveDir * baseSpeed * moveMagnitude
			local lerpAlpha = 1 - math.exp(-acceleration * dt)
			horizontalVelocity = horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
		else
			local lerpAlpha = 1 - math.exp(-deceleration * dt)
			horizontalVelocity = horizontalVelocity:Lerp(Vector3.zero, lerpAlpha)
		end
	else
		currentVerticalVelocity = math.max(currentVerticalVelocity - gravityForce * dt, terminalVelocity)

		if moveMagnitude > 0.1 then
			local worldMoveDir = lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit
			local targetVelocity = worldMoveDir * baseSpeed * AIR_CONTROL * -(currentVerticalVelocity / airResistance)
			local lerpAlpha = 1 - math.exp(-acceleration * 0.3 * dt)
			horizontalVelocity = horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
		else
			horizontalVelocity *= (1 - airResistance * 0.5 * dt)
		end
	end

	currentVelocity = Vector3.new(horizontalVelocity.X, currentVerticalVelocity, horizontalVelocity.Z)

	-- // Update UI
	if VectorForces then
		VectorForces.X.Text = "X: "..tostring(math.round(horizontalVelocity.X))
		VectorForces.Y.Text = "Y: "..tostring(math.round(currentVerticalVelocity))
		VectorForces.Z.Text = "Z: "..tostring(math.round(horizontalVelocity.Z))
	end

	local camCFrame = cam.CFrame
	local targetLook = Vector3.new(camCFrame.LookVector.X, 0, camCFrame.LookVector.Z).Unit
	lastCFrame = lastCFrame:Lerp(CFrame.new(rigBody.Position, rigBody.Position + targetLook), 1 - math.exp(-turnSpeed * dt))
end

-- // Network updates with throttling and delta compression for optomization
local function sendNetworkUpdate()
	local currentTime = tick()

	-- // Check if its time to send an update
	if currentTime - lastSentTime < sendInterval then
		return
	end

	-- // Check if theres been a significant change in velocity or position
	local velocityDelta = (currentVelocity - lastSentVelocity).Magnitude
	local positionDelta = (lastCFrame.Position - lastSentCFrame.Position).Magnitude
	local positionChanged = positionDelta > 0.01  -- // Small threshold to account for floating point precision
	
	--[[
	if velocityDelta > movementThreshold or positionChanged then
		-- // Send the update to the server, we have to do this since velocity changes cant be applied locally
		moveRigEvent:FireServer(currentVelocity, lastCFrame) -- // currentVelocity: LinearVelocity. lastCFrame: AngularVelocity.

		-- // Update the last sent values
		lastSentVelocity = currentVelocity
		lastSentCFrame = lastCFrame
		lastSentTime = currentTime
	end
	--]]
	
	moveRigEvent:FireServer(currentVelocity, lastCFrame) -- // currentVelocity: LinearVelocity. lastCFrame: AngularVelocity.

	-- // Update the last sent values
	lastSentVelocity = currentVelocity
	lastSentCFrame = lastCFrame
	lastSentTime = currentTime
end

-- // Main update loop
local function onRenderStepped(dt)
	updatePhysics(dt)
	sendNetworkUpdate()
end

-- // Jump handling
local function onJumpRequest()
	if isGrounded and canJump and not isSliding then
		currentVerticalVelocity = jumpPower
		canJump = false
	end
end

-- // Character reset
local function onKeyPress(input, gameProcessed)
	if not gameProcessed and input.UserInputType == Enum.UserInputType.Keyboard and input.KeyCode == Enum.KeyCode.R then
		humanoid.Health = 0
	end
end

-- // Character added handling so we can just reset the real character to bring the fake rig back
local function CharacterAdded(newChar)
	-- // Clean up old connections
	cleanupConnections()

	-- // Set up new character
	local newRig = workspace:WaitForChild("Rig_" .. player.Name)
	rigHead = newRig:WaitForChild("Head")
	rigBody = newRig:WaitForChild("Body")
	cam.CameraSubject = rigHead
	character = newChar
	humanoid = newChar:WaitForChild("Humanoid")
	humanoidRootPart = newChar:WaitForChild("HumanoidRootPart")

	-- // Reset tracking variables
	currentVelocity = Vector3.zero
	currentVerticalVelocity = 0
	lastCFrame = CFrame.new()
	isGrounded = false
	canJump = true
	isSliding = false
	lastSentTime = 0
	lastSentVelocity = Vector3.zero
	lastSentCFrame = CFrame.new()

	raycastParams.FilterDescendantsInstances = {rigHead, rigBody, character}

	-- // Set up new connections
	table.insert(connections, RunService.RenderStepped:Connect(onRenderStepped))
	table.insert(connections, UserInputService.JumpRequest:Connect(onJumpRequest))
	table.insert(connections, UserInputService.InputBegan:Connect(onKeyPress))
end

local function PlayerAdded(plr)
	if plr ~= player then return end

	plr.CharacterAdded:Connect(CharacterAdded)

	local char = plr.Character
	if char then
		CharacterAdded(char)
	end
end

-- // Set up initial connections
table.insert(connections, RunService.RenderStepped:Connect(onRenderStepped))
table.insert(connections, UserInputService.JumpRequest:Connect(onJumpRequest))
table.insert(connections, UserInputService.InputBegan:Connect(onKeyPress))
table.insert(connections, player.CharacterAdded:Connect(CharacterAdded))
Players.PlayerAdded:Connect(PlayerAdded)

script.Destroying:Connect(cleanupConnections)

for _, plr in pairs(Players:GetPlayers()) do
	PlayerAdded(plr)
end
