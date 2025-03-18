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
local AIR_CONTROL = 13  -- // Control factor while in air
local SLIDE_FRICTION = 0.05 -- // Low friction for sliding
local SLIDE_THRESHOLD = 45  -- // Degrees; slope angle to trigger sliding

-- // Network optimization
local sendInterval = 1/60  -- // How fast the client updates the server
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
			cachedGroundInfo = {true, slopeAngle <= SLIDE_THRESHOLD, slopeAngle} -- // Set isGrounded to true, set canStand if the slopeAngle were on is less than or equal to SLIDE_THRESHOLD and set the slopeAngle
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
-- // Physics calculations
---------------------------------------------

local function updatePhysics(dt)
	local moveDir = humanoid.MoveDirection
	local moveMagnitude = moveDir.Magnitude
	local grounded, canStand, slopeAngle = checkGround() -- // Set variables from checkGround()
	local justLanded = grounded and not isGrounded
	isGrounded = grounded
	isSliding = grounded and not canStand and slopeAngle > SLIDE_THRESHOLD
	local horizontalVelocity = Vector3.new(currentVelocity.X, 0, currentVelocity.Z) -- // Currently (0, 0, 0)

	if grounded then
		if justLanded or currentVerticalVelocity < 0 then -- // So we cant just jump infinetly while mid air
			currentVerticalVelocity = 0
			canJump = true
		end
		
		-- // Setting horizontalVelocity and lerping it so theres acceleration / deceleration only if we are grounded
		
		if isSliding then -- // Accelerate the horizontalVelocity while sliding
			local slopeDir = Vector3.new(horizontalVelocity.X, 0, horizontalVelocity.Z).Unit
			if slopeDir.Magnitude == 0 then slopeDir = -lastCFrame.LookVector end 
			horizontalVelocity += slopeDir * (gravityForce * math.sin(math.rad(slopeAngle)) * dt)
			horizontalVelocity *= (1 - 0.1 * dt)
		elseif moveMagnitude > 0.1 then -- // Although the Rig and the actual character are 2 seperate things, you can still use Humanoid.MoveDirection.Magnitude even if the player cant move (eg. humanoid.WalkSpeed = 0)
			local worldMoveDir = lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit
			local targetVelocity = worldMoveDir * baseSpeed * moveMagnitude
			local lerpAlpha = 1 - math.exp(-acceleration * dt)
			horizontalVelocity = horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
		else -- // If we are not moving, then decelerate
			local lerpAlpha = 1 - math.exp(-deceleration * dt)
			horizontalVelocity = horizontalVelocity:Lerp(VECTOR_ZERO, lerpAlpha)
		end
	else
		
		-- // We can adjust the vertical velocity if we are not grounded, allowing for gravity and also a termainal velocity so we accelerate downward when falling
		
		currentVerticalVelocity = math.max(currentVerticalVelocity - gravityForce * dt, terminalVelocity)
		
		-- // New equation for accelerating while midair 
		
		if moveMagnitude > 0.1 then
			local worldMoveDir = lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit
			local airControlFactor = math.max(0, -currentVerticalVelocity / airResistance)
			local targetVelocity = worldMoveDir * baseSpeed * AIR_CONTROL * airControlFactor
			local lerpAlpha = 1 - math.exp(-acceleration * 0.3 * dt)
			horizontalVelocity = horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
		else -- // Triggers while we are not moving
			horizontalVelocity *= (1 - 20 * dt)
		end
	end

	-- // All of the calculations together, clamping the X and Z axis based on the horizontalTerminalVelocity.
	currentVelocity = Vector3.new(
		math.clamp(horizontalVelocity.X, -horizontalTerminalVelocity, horizontalTerminalVelocity), -- // X
		currentVerticalVelocity, -- // Y
		math.clamp(horizontalVelocity.Z, -horizontalTerminalVelocity, horizontalTerminalVelocity) -- // Z
	)
	
	
	-- // For UI Updates
	
	local currentTime = tick()

	if VectorForces and currentTime - lastUiUpdateTime >= uiUpdateInterval then
		VectorForces.X.Text = "X: " .. tostring(math.round(horizontalVelocity.X))
		VectorForces.Y.Text = "Y: " .. tostring(math.round(currentVerticalVelocity))
		VectorForces.Z.Text = "Z: " .. tostring(math.round(horizontalVelocity.Z))
		lastUiUpdateTime = currentTime
	end

	-- // Since the rig and the actual character are 2 different things, you cant use things like the HRP's LookVector as the character needs to be able to move doing that. Instead, we can use the cameras LookVector which works anywhere. This allows for a shiftlock-like camera where the rig faces where you look
	
	local camCFrame = cam.CFrame
	local targetLook = Vector3.new(camCFrame.LookVector.X, 0, camCFrame.LookVector.Z).Unit
	lastCFrame = lastCFrame:Lerp(CFrame.new(rigBody.Position, rigBody.Position + targetLook), 1 - math.exp(-turnSpeed * dt))
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
