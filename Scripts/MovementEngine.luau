-- // Everything in this code could obviously be much more optomized like using GroundControllers, Character Controllers, etc. This purpose of this was to create a whole movement engine from scratch only using Robloxs physics including gravity, movement, etc. WIP.

---------------------------------------------
-- // Variables
---------------------------------------------

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
local acceleration = 5  -- // Rate of speed increase
local deceleration = 4  -- // Rate of speed decrease
local turnSpeed = 8  -- // Speed of rotation
local gravityForce = workspace.Gravity  -- // Gravity strength from workspace
local groundCheckDistance = 2.5  -- // Distance to check for ground
local terminalVelocity = -1000  -- // Maximum fall speed
local horizontalTerminalVelocity = 1500 -- // Maximum horizontal speed
local airResistance = 80  -- // Resistance when in air
local jumpPower = 40  -- // Force of jump
local AirControl = 13  -- // Control factor while in air
local SlideThreshold = 45  -- // Degrees; slope angle to trigger sliding

-- // Network optimization
local sendInterval = 1/30  -- // How fast the client updates the server
local lastSentTime = 0
local lastSentVelocity = Vector3.zero
local lastSentCFrame = CFrame.new()

-- // Performance measures
local groundCheckInterval = 1/30  -- // How fast it raycasts to the ground for gravity
local lastGroundCheckTime = 0
local cachedGroundInfo = {false, false, 0}  -- // {isGrounded, canStand, slopeAngle}

-- // Checks
local currentVelocity = Vector3.zero  -- // Current movement velocity
local currentVerticalVelocity = 0  -- // Vertical number of velocity
local lastCFrame = CFrame.new()  -- // Last frame's CFrame for lerping
local isGrounded = false  -- // Whether character is on ground
local canJump = true  -- // Whether character can jump
local isSliding = false  -- // Whether character is sliding
local raycastParams = RaycastParams.new()  
raycastParams.FilterType = Enum.RaycastFilterType.Exclude 
raycastParams.FilterDescendantsInstances = {rigHead, rigBody, character}
local VECTOR_ZERO = Vector3.zero

-- // Memory management
local connections = {}

-- // UI updates
local uiUpdateInterval = 0.1
local lastUiUpdateTime = 0

---------------------------------------------
-- // Helper Functions
---------------------------------------------

local function checkGround() -- // Since you cant really apply gravity when theres a LinearVelocity acting on the X or Z axis, we need to constantly raycast to the ground and see if we should apply the gravity (Set at a fixed rate for optomization, see the groundCheckInterval to change it)
	local currentTime = tick()
	if currentTime - lastGroundCheckTime >= groundCheckInterval then
		local raycastResult = workspace:Raycast(rigBody.Position, Vector3.new(0, -groundCheckDistance, 0), raycastParams)
		if raycastResult then
			local normal = raycastResult.Normal
			local slopeAngle = math.deg(math.acos(normal.Y)) -- // Since there is a sliding mechanic, we can see the angle of the slope at which we are on so we just dont slide on every surface
			cachedGroundInfo = {true, slopeAngle <= SlideThreshold, slopeAngle} -- // Set isGrounded to true, set canStand if the slopeAngle were on is less than or equal to SLIDE_THRESHOLD and set the slopeAngle
			print(slopeAngle)
		else
			cachedGroundInfo = {false, false, 0} -- // If no Raycast Result
		end
		lastGroundCheckTime = currentTime -- // Reset lastGroundCheckTime so it loops infinetly based on groundCheckInterval
	end
	return unpack(cachedGroundInfo) -- // Return the unpacked version of the cachedGroundInfo table
end

local function onJumpRequest()
	if isGrounded and canJump and not isSliding then
		currentVerticalVelocity = jumpPower
		canJump = false
	end
end

local function cleanupConnections() -- // To prevent memory leaks from happening, all Connections are stored in a table
	for _, connection in pairs(connections) do
		if connection.Connected then
			connection:Disconnect()
		end
	end
	connections = {}
end

local function sendNetworkUpdate()
	local currentTime = tick()
	if currentTime - lastSentTime < sendInterval then
		return
	end

	moveRigEvent:FireServer(currentVelocity, lastCFrame)
	lastSentVelocity = currentVelocity
	lastSentCFrame = lastCFrame
	lastSentTime = currentTime
end

---------------------------------------------
-- // Physics Calculations
---------------------------------------------

-- // Calculates slope-based sliding velocity
local function calculateSlideVelocity(horizontalVelocity, slopeAngle, dt)
	local slopeDir = Vector3.new(horizontalVelocity.X, 0, horizontalVelocity.Z).Unit
	if slopeDir.Magnitude == 0 then 
		slopeDir = -lastCFrame.LookVector 
	end

	-- // Calculate acceleration due to slope using gravity and angle
	-- // sin(slopeAngle) gives us the component of gravity along the slope
	local slopeAcceleration = gravityForce * math.sin(math.rad(slopeAngle)) * dt
	local slideVelocity = horizontalVelocity + slopeDir * slopeAcceleration

	-- // Apply basic friction reduction
	slideVelocity *= (1 - 0.1 * dt)
	return slideVelocity
end

-- // Handles ground movement acceleration/deceleration
local function calculateGroundMovement(moveDir, horizontalVelocity, dt)
	local moveMagnitude = moveDir.Magnitude
	if moveMagnitude > 0.1 then
		-- // Convert world direction to local space and normalize
		local worldMoveDir = lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit
		local targetVelocity = worldMoveDir * baseSpeed * moveMagnitude

		-- // Exponential interpolation for smooth acceleration, this allows us to accelerate and decelerate
		local lerpAlpha = 1 - math.exp(-acceleration * dt)
		return horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
	else
		-- // Exponential deceleration when not moving
		local lerpAlpha = 1 - math.exp(-deceleration * dt)
		return horizontalVelocity:Lerp(VECTOR_ZERO, lerpAlpha)
	end
end

-- // Handles air movement with control factor
local function calculateAirMovement(moveDir, horizontalVelocity, dt)
	local moveMagnitude = moveDir.Magnitude
	if moveMagnitude > 0.1 then
		local worldMoveDir = lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit

		-- // Calculate air control based on fall speed
		-- // Faster falling = less control
		local airControlFactor = math.max(0, -currentVerticalVelocity / airResistance)
		local targetVelocity = worldMoveDir * baseSpeed * AirControl * airControlFactor

		-- // Reduced acceleration in air (0.3x normal)
		local lerpAlpha = 1 - math.exp(-acceleration * 0.3 * dt)
		return horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
	else
		-- // Simple air resistance when not moving
		return horizontalVelocity * (1 - 20 * dt)
	end
end

-- // Updates UI with velocity information
local function updateVelocityUI(horizontalVelocity)
	local currentTime = tick()
	if VectorForces and currentTime - lastUiUpdateTime >= uiUpdateInterval then
		VectorForces.X.Text = "X: " .. tostring(math.round(horizontalVelocity.X))
		VectorForces.Y.Text = "Y: " .. tostring(math.round(currentVerticalVelocity))
		VectorForces.Z.Text = "Z: " .. tostring(math.round(horizontalVelocity.Z))
		lastUiUpdateTime = currentTime
	end
end

-- // Updates character rotation based on camera
local function updateRotation(dt)
	local camCFrame = cam.CFrame
	local targetLook = Vector3.new(camCFrame.LookVector.X, 0, camCFrame.LookVector.Z).Unit

	-- // Smoothly interpolate rotation using exponential decay
	local lerpAlpha = 1 - math.exp(-turnSpeed * dt)
	lastCFrame = lastCFrame:Lerp(CFrame.new(rigBody.Position, rigBody.Position + targetLook), lerpAlpha)
end

---------------------------------------------
-- // Main Physics Function
---------------------------------------------

local function updatePhysics(dt)
	local moveDir = humanoid.MoveDirection
	local grounded, canStand, slopeAngle = checkGround()
	local justLanded = grounded and not isGrounded
	isGrounded = grounded
	isSliding = grounded and not canStand and slopeAngle > SlideThreshold
	local horizontalVelocity = Vector3.new(currentVelocity.X, 0, currentVelocity.Z)

	if grounded then
		if justLanded or currentVerticalVelocity < 0 then
			currentVerticalVelocity = 0
			canJump = true
		end

		if isSliding then
			horizontalVelocity = calculateSlideVelocity(horizontalVelocity, slopeAngle, dt)
		else
			horizontalVelocity = calculateGroundMovement(moveDir, horizontalVelocity, dt)
		end
	else
		-- // Apply gravity with terminal velocity cap
		currentVerticalVelocity = math.max(currentVerticalVelocity - gravityForce * dt, terminalVelocity)
		horizontalVelocity = calculateAirMovement(moveDir, horizontalVelocity, dt)
	end

	-- // Combine horizontal and vertical velocities with clamping
	currentVelocity = Vector3.new(
		math.clamp(horizontalVelocity.X, -horizontalTerminalVelocity, horizontalTerminalVelocity),
		currentVerticalVelocity,
		math.clamp(horizontalVelocity.Z, -horizontalTerminalVelocity, horizontalTerminalVelocity)
	)

	updateVelocityUI(horizontalVelocity)
	updateRotation(dt)
end

---------------------------------------------
-- // Inputs and Cleanup
---------------------------------------------

local function onRenderStepped(dt)
	updatePhysics(dt)
	sendNetworkUpdate()
end


local function onKeyPress(input, gameProcessed)
	if not gameProcessed and input.UserInputType == Enum.UserInputType.Keyboard and input.KeyCode == Enum.KeyCode.R then
		humanoid.Health = 0
	end
end

-- // Basic cleanup when the character is added

local function CharacterAdded(newChar)
	cleanupConnections()
	local newRig = workspace:WaitForChild("Rig_" .. player.Name)
	rigHead = newRig:WaitForChild("Head")
	rigBody = newRig:WaitForChild("Body")
	cam.CameraSubject = rigHead
	character = newChar
	humanoid = newChar:WaitForChild("Humanoid")
	humanoidRootPart = newChar:WaitForChild("HumanoidRootPart")
	currentVelocity = VECTOR_ZERO
	currentVerticalVelocity = 0
	lastCFrame = CFrame.new()
	isGrounded = false
	canJump = true
	isSliding = false
	lastSentTime = 0
	lastSentVelocity = VECTOR_ZERO
	lastSentCFrame = CFrame.new()
	raycastParams.FilterDescendantsInstances = {rigHead, rigBody, character}
	
	task.wait(0.1) -- // Apply the connections after a short delay since after you respawn there may be some forces that just fling you
	
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

-- // Insert all of the inputs and loops

table.insert(connections, RunService.RenderStepped:Connect(onRenderStepped))
table.insert(connections, UserInputService.JumpRequest:Connect(onJumpRequest))
table.insert(connections, UserInputService.InputBegan:Connect(onKeyPress))
table.insert(connections, player.CharacterAdded:Connect(CharacterAdded))

Players.PlayerAdded:Connect(PlayerAdded)

script.Destroying:Connect(cleanupConnections)

for _, plr in pairs(Players:GetPlayers()) do
	PlayerAdded(plr)
end
