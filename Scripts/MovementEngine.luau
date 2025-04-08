-- // Everything in this code could obviously be much more optimized like using GroundControllers, Character Controllers, etc.
-- // The purpose of this was to create a whole movement engine from scratch only using Roblox's physics including gravity, movement, etc. WIP.

---------------------------------------------
-- // Variables
---------------------------------------------

-- // Services and References
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")
local UserInputService = game:GetService("UserInputService")
local moveRigEvent = ReplicatedStorage:WaitForChild("Events"):WaitForChild("MoveRigEvent")
local cam = workspace.CurrentCamera
local player = Players.LocalPlayer
local character = player.Character or player.CharacterAdded:Wait()
local humanoid = character:WaitForChild("Humanoid")
local humanoidRootPart = character:WaitForChild("HumanoidRootPart")
local rigHead = workspace:WaitForChild("Rig_" .. player.Name):WaitForChild("Head")
local rigBody = workspace:WaitForChild("Rig_" .. player.Name):WaitForChild("Body")
local player_Gui = player:WaitForChild("PlayerGui")
local vectorForces = player_Gui:WaitForChild("Keys"):WaitForChild("VectorForces")
local BitBuffer = require(ReplicatedStorage.Modules.BitBuffer)

-- // Movement Variables
local MovementVariables = {
    baseSpeed = 30,          -- // Base movement speed
    acceleration = 3,        -- // Rate of speed increase
    deceleration = 4,        -- // Rate of speed decrease
    turnSpeed = 8,           -- // Speed of rotation
    gravityForce = workspace.Gravity,  -- // Gravity strength from workspace
    groundCheckDistance = 2.5,  -- // Distance to check for ground
    terminalVelocity = -1000,   -- // Maximum fall speed
    horizontalTerminalVelocity = 1500,  -- // Maximum horizontal speed
    airResistance = 80,         -- // Resistance when in air
    jumpPower = 80,            -- // Force of jump
    AirControl = 13,           -- // Control factor while in air
	SlideThreshold = 45,        -- // Degrees; slope angle to trigger sliding
	downVector = Vector3.new(0, -1, 0) -- // Just a cached down vector for optomization for gravity simulation. Don't change.
}

MovementVariables.groundCheckDirection = MovementVariables.downVector * MovementVariables.groundCheckDistance -- // We cant add this to the MovementVariables table since it wont be able to access the downVector while inside of it.

-- // Network optimization
local NetworkVariables = {
    sendInterval = 1/60,       -- // How fast the client updates the server
    lastSentTime = 0,
    lastSentVelocity = Vector3.zero,
    lastSentCFrame = CFrame.new()
}

-- // Performance measures
local PerformanceVariables = {
    groundCheckInterval = 1/30,  -- // How fast it raycasts to the ground for gravity
    lastGroundCheckTime = 0,
    cachedGroundInfo = {false, false, 0}  -- // {isGrounded, canStand, slopeAngle}
}

-- // Checks
local MiscVariables = {
    currentVelocity = Vector3.zero,    -- // Current movement velocity
    currentVerticalVelocity = 0,       -- // Vertical number of velocity
    lastCFrame = CFrame.new(),         -- // Last frame's CFrame for lerping
    isGrounded = false,                -- // Whether character is on ground
    canJump = true,                    -- // Whether character can jump
    isSliding = false,                 -- // Whether character is sliding
    raycastParams = RaycastParams.new(),
    VECTOR_ZERO = Vector3.zero
}

MiscVariables.raycastParams.FilterType = Enum.RaycastFilterType.Exclude
MiscVariables.raycastParams.FilterDescendantsInstances = {rigHead, rigBody, character}

-- // Memory management
local connections = {}

-- // UI updates
local UIUpdates = {
    uiUpdateInterval = 0.01,
    lastUiUpdateTime = 0
}

---------------------------------------------
-- // Helper Functions
---------------------------------------------

local function checkGround()
    local currentTime = tick()
    if currentTime - PerformanceVariables.lastGroundCheckTime >= PerformanceVariables.groundCheckInterval then
		local raycastResult = workspace:Raycast(
			rigBody.Position, 
			MovementVariables.groundCheckDirection, 
			MiscVariables.raycastParams
		)
        if raycastResult then
            local normal = raycastResult.Normal
            local slopeAngle = math.deg(math.acos(normal.Y))
            PerformanceVariables.cachedGroundInfo = {true, slopeAngle <= MovementVariables.SlideThreshold, slopeAngle}
        else
            PerformanceVariables.cachedGroundInfo = {false, false, 0}
        end
        PerformanceVariables.lastGroundCheckTime = currentTime
    end
    return unpack(PerformanceVariables.cachedGroundInfo)
end

local function onJumpRequest()
    if MiscVariables.isGrounded and MiscVariables.canJump and not MiscVariables.isSliding then
        MiscVariables.currentVerticalVelocity = MovementVariables.jumpPower
        MiscVariables.canJump = false
    end
end

local function cleanupConnections()
    for _, connection in pairs(connections) do
        if connection.Connected then
            connection:Disconnect()
        end
    end
    connections = {}
end

---------------------------------------------
-- // Bitpacking Methods (So we can reduce the amount of information sent to the server, around a ~75%(?) reduction of network bandwidth)
---------------------------------------------

local VELOCITY_OFFSET_X = 3000
local VELOCITY_OFFSET_Y = 2000
local VELOCITY_OFFSET_Z = 3000
local POSITION_OFFSET = 409.6
local POSITION_SCALE = 0.1
local YAW_SCALE = 512 / (2 * math.pi)

local POSITION_FACTOR = 1 / POSITION_SCALE

local function packData(velocity, cframe)
	local buffer = BitBuffer.Create()

	-- // Velocity packing (38 bits)
	buffer:WriteUnsigned(13, math.clamp(math.round(velocity.X + VELOCITY_OFFSET_X), 0, 6000))
	buffer:WriteUnsigned(13, math.clamp(math.round(velocity.Z + VELOCITY_OFFSET_Z), 0, 6000))
	buffer:WriteUnsigned(12, math.clamp(math.round(velocity.Y + VELOCITY_OFFSET_Y), 0, 3000))

	-- // Position packing (39 bits)
	local pos = cframe.Position
	buffer:WriteUnsigned(13, math.clamp(math.round((pos.X + POSITION_OFFSET) * POSITION_FACTOR), 0, 8191))
	buffer:WriteUnsigned(13, math.clamp(math.round((pos.Y + POSITION_OFFSET) * POSITION_FACTOR), 0, 8191))
	buffer:WriteUnsigned(13, math.clamp(math.round((pos.Z + POSITION_OFFSET) * POSITION_FACTOR), 0, 8191))

	-- // Yaw packing (9 bits)
	local cf = cframe
	local yaw = math.atan2(-cf.RightVector.Z, cf.RightVector.X)
	buffer:WriteUnsigned(9, math.clamp(math.round((yaw + math.pi) * YAW_SCALE), 0, 511))

	return buffer:ToBase64()
end
---------------------------------------------
-- // Physics Calculations
---------------------------------------------

local function calculateSlideVelocity(horizontalVelocity, slopeAngle, dt)
    local slopeDir = Vector3.new(horizontalVelocity.X, 0, horizontalVelocity.Z).Unit
    if slopeDir.Magnitude == 0 then 
        slopeDir = -MiscVariables.lastCFrame.LookVector 
    end

	local sinSlopeAngle = math.sin(math.rad(slopeAngle))
	local slopeAcceleration = MovementVariables.gravityForce * sinSlopeAngle * dt
    local slideVelocity = horizontalVelocity + slopeDir * slopeAcceleration
    slideVelocity = slideVelocity * (1 - 0.1 * dt)
    return slideVelocity
end

local function calculateGroundMovement(moveDir, horizontalVelocity, dt)
    local moveMagnitude = moveDir.Magnitude
    if moveMagnitude > 0.1 then
        local worldMoveDir = MiscVariables.lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit
        local targetVelocity = worldMoveDir * MovementVariables.baseSpeed * moveMagnitude
		local lerpAlpha = 1 - math.exp(-MovementVariables.acceleration * dt)
        return horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
    else
        local lerpAlpha = 1 - math.exp(-MovementVariables.deceleration * dt)
        return horizontalVelocity:Lerp(MiscVariables.VECTOR_ZERO, lerpAlpha)
    end
end

local function calculateAirMovement(moveDir, horizontalVelocity, dt)
    local moveMagnitude = moveDir.Magnitude
    if moveMagnitude > 0.1 then
        local worldMoveDir = MiscVariables.lastCFrame:VectorToObjectSpace(Vector3.new(moveDir.X, 0, moveDir.Z)).Unit
        local airControlFactor = math.max(0, -MiscVariables.currentVerticalVelocity / MovementVariables.airResistance)
        local targetVelocity = worldMoveDir * MovementVariables.baseSpeed * MovementVariables.AirControl * airControlFactor
        local lerpAlpha = 1 - math.exp(-MovementVariables.acceleration * 0.3 * dt) 
        return horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
    else
        return horizontalVelocity * (1 - 20 * dt)
    end
end

local function updateVelocityUI(horizontalVelocity)
    local currentTime = tick()
    if vectorForces and currentTime - UIUpdates.lastUiUpdateTime >= UIUpdates.uiUpdateInterval then
        vectorForces.X.Text = "X: " .. tostring(math.round(horizontalVelocity.X))
        vectorForces.Y.Text = "Y: " .. tostring(math.round(MiscVariables.currentVerticalVelocity))
        vectorForces.Z.Text = "Z: " .. tostring(math.round(horizontalVelocity.Z))
        UIUpdates.lastUiUpdateTime = currentTime
    end
end

local function updateRotation(dt)
	local camCFrame = cam.CFrame
	local targetLook = Vector3.new(camCFrame.LookVector.X, 0, camCFrame.LookVector.Z).Unit
	local lerpAlpha = MovementVariables.turnSpeed * dt * 2
	MiscVariables.lastCFrame = MiscVariables.lastCFrame:Lerp(CFrame.new(rigBody.Position, rigBody.Position + targetLook), math.clamp(lerpAlpha, 0, 1))
end

---------------------------------------------
-- // Main Physics Function
---------------------------------------------

local function updatePhysics(dt)
    local moveDir = humanoid.MoveDirection
    local grounded, canStand, slopeAngle = checkGround()
    local justLanded = grounded and not MiscVariables.isGrounded
    MiscVariables.isGrounded = grounded
    MiscVariables.isSliding = grounded and slopeAngle >= MovementVariables.SlideThreshold
    local horizontalVelocity = Vector3.new(MiscVariables.currentVelocity.X, 0, MiscVariables.currentVelocity.Z)

    if grounded then
        if justLanded or MiscVariables.currentVerticalVelocity < 0 then
            MiscVariables.currentVerticalVelocity = 0
            MiscVariables.canJump = true
        end

        if MiscVariables.isSliding then
            horizontalVelocity = calculateSlideVelocity(horizontalVelocity, slopeAngle, dt)
        else
            horizontalVelocity = calculateGroundMovement(moveDir, horizontalVelocity, dt)
        end
    else
        MiscVariables.currentVerticalVelocity = math.max(MiscVariables.currentVerticalVelocity - MovementVariables.gravityForce * dt, MovementVariables.terminalVelocity)
        horizontalVelocity = calculateAirMovement(moveDir, horizontalVelocity, dt)
    end

    MiscVariables.currentVelocity = Vector3.new(
        math.clamp(horizontalVelocity.X, -MovementVariables.horizontalTerminalVelocity, MovementVariables.horizontalTerminalVelocity),
        MiscVariables.currentVerticalVelocity,
        math.clamp(horizontalVelocity.Z, -MovementVariables.horizontalTerminalVelocity, MovementVariables.horizontalTerminalVelocity)
    )

    updateVelocityUI(horizontalVelocity)
    updateRotation(dt)
end

local function sendNetworkUpdate()
	local currentTime = tick()
	if currentTime - NetworkVariables.lastSentTime < NetworkVariables.sendInterval then
		return
	end

	local packedData = packData(MiscVariables.currentVelocity, MiscVariables.lastCFrame)
	
	moveRigEvent:FireServer(packedData)
	
	NetworkVariables.lastSentVelocity = MiscVariables.currentVelocity
	NetworkVariables.lastSentCFrame = MiscVariables.lastCFrame
	NetworkVariables.lastSentTime = currentTime
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

local function CharacterAdded(newChar)
    cleanupConnections()
    local newRig = workspace:WaitForChild("Rig_" .. player.Name)
    rigHead = newRig:WaitForChild("Head")
    rigBody = newRig:WaitForChild("Body")
    cam.CameraSubject = rigHead
    character = newChar
    humanoid = newChar:WaitForChild("Humanoid")
    humanoidRootPart = newChar:WaitForChild("HumanoidRootPart")
    MiscVariables.currentVelocity = MiscVariables.VECTOR_ZERO
    MiscVariables.currentVerticalVelocity = 0
    MiscVariables.lastCFrame = CFrame.new()
    MiscVariables.isGrounded = false
    MiscVariables.canJump = true
    MiscVariables.isSliding = false
    NetworkVariables.lastSentTime = 0
    NetworkVariables.lastSentVelocity = MiscVariables.VECTOR_ZERO
    NetworkVariables.lastSentCFrame = CFrame.new()
    MiscVariables.raycastParams.FilterDescendantsInstances = {rigHead, rigBody, character}
    
    table.insert(connections, RunService.RenderStepped:Connect(onRenderStepped))
    table.insert(connections, UserInputService.JumpRequest:Connect(onJumpRequest))
    table.insert(connections, UserInputService.InputBegan:Connect(onKeyPress))
end

local function PlayerAdded(plr)
	if plr ~= player then return end

	local charAddedConn = plr.CharacterAdded:Connect(CharacterAdded)
	table.insert(connections, charAddedConn)

	local char = plr.Character
	if char then
		CharacterAdded(char)
	end
end

-- // Connect everything
table.insert(connections, RunService.RenderStepped:Connect(onRenderStepped))
table.insert(connections, UserInputService.JumpRequest:Connect(onJumpRequest))
table.insert(connections, UserInputService.InputBegan:Connect(onKeyPress))
table.insert(connections, player.CharacterAdded:Connect(CharacterAdded))
table.insert(connections, Players.PlayerAdded:Connect(PlayerAdded))

for _, plr in pairs(Players:GetPlayers()) do
	PlayerAdded(plr)
end

script.Destroying:Connect(cleanupConnections)
