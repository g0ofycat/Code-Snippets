--[[
    Custom Physics Movement System for Roblox
    
    A comprehensive movement engine built from scratch using Roblox's physics system.
    Features include custom gravity, ground detection, sliding mechanics, air control,
    and network optimization with bit packing.
    
    Note: This could be optimized further with GroundControllers, Character Controllers, etc.
    The purpose is to demonstrate a complete movement system using only Roblox physics.
--]]

-- ===============================================
-- SERVICES AND DEPENDENCIES
-- ===============================================

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local Players = game:GetService("Players")
local UserInputService = game:GetService("UserInputService")

local BitBuffer = require(ReplicatedStorage.Modules.BitBuffer)

local moveRigEvent = ReplicatedStorage:WaitForChild("Events"):WaitForChild("MoveRigEvent")

local player = Players.LocalPlayer
local cam = workspace.CurrentCamera

-- ===============================================
-- CONFIGURATION
-- ===============================================

local Config = {
	Movement = {
		baseSpeed = 30,
		acceleration = 3,
		deceleration = 4,
		turnSpeed = 8,
		jumpPower = 80,
		airControl = 13,
		slideThreshold = 45, -- // In Degrees
		horizontalTerminalVelocity = 1500,
		terminalVelocity = -1000,
		airResistance = 80,
		gravityForce = workspace.Gravity,
		groundCheckDistance = 2.5,
		downVector = Vector3.new(0, -1, 0)
	},

	Network = {
		sendInterval = 1/60,
		velocityThreshold = 0.1,
		positionThreshold = 0.1
	},

	Performance = {
		groundCheckInterval = 1/30,
		uiUpdateInterval = 0.01
	},

	BitPacking = {
		velocityOffsetX = 3000,
		velocityOffsetY = 2000,
		velocityOffsetZ = 3000,
		positionOffset = 409.6,
		positionScale = 0.1,
		yawScale = 512 / (2 * math.pi)
	}
}

Config.Movement.groundCheckDirection = Config.Movement.downVector * Config.Movement.groundCheckDistance
Config.BitPacking.positionFactor = 1 / Config.BitPacking.positionScale

-- ===============================================
-- STATE MANAGEMENT
-- ===============================================

local State = {
	character = nil,
	humanoid = nil,
	humanoidRootPart = nil,
	rigHead = nil,
	rigBody = nil,
	playerGui = nil,
	vectorForcesUI = nil,

	currentVelocity = Vector3.zero,
	currentVerticalVelocity = 0,
	lastCFrame = CFrame.new(),
	isGrounded = false,
	canJump = true,
	isSliding = false,

	cachedGroundInfo = {false, false, 0},
	lastGroundCheckTime = 0,
	lastUiUpdateTime = 0,

	lastSentTime = 0,
	lastSentVelocity = Vector3.zero,
	lastSentCFrame = CFrame.new(),

	raycastParams = RaycastParams.new(),

	connections = {},
	
	VECTOR_ZERO = Vector3.zero
}

State.raycastParams.FilterType = Enum.RaycastFilterType.Exclude

-- ===============================================
-- UTILITY FUNCTIONS
-- ===============================================

local Utils = {}

function Utils.cleanupConnections()
	for _, connection in pairs(State.connections) do
		if connection and connection.Connected then
			connection:Disconnect()
		end
	end
	State.connections = {}
end

function Utils.addConnection(connection)
	table.insert(State.connections, connection)
	return connection
end

function Utils.updateRaycastFilter()
	State.raycastParams.FilterDescendantsInstances = {
		State.rigHead, 
		State.rigBody, 
		State.character
	}
end

function Utils.resetMovementState()
	State.currentVelocity = State.VECTOR_ZERO
	State.currentVerticalVelocity = 0
	State.lastCFrame = CFrame.new()
	State.isGrounded = false
	State.canJump = true
	State.isSliding = false
	State.lastSentTime = 0
	State.lastSentVelocity = State.VECTOR_ZERO
	State.lastSentCFrame = CFrame.new()
	State.cachedGroundInfo = {false, false, 0}
end

-- ===============================================
-- GROUND DETECTION SYSTEM
-- ===============================================

local GroundDetection = {}

function GroundDetection.checkGround()
	local currentTime = tick()

	if currentTime - State.lastGroundCheckTime < Config.Performance.groundCheckInterval then
		return unpack(State.cachedGroundInfo)
	end

	local raycastResult = workspace:Raycast(
		State.rigBody.Position,
		Config.Movement.groundCheckDirection,
		State.raycastParams
	)

	if raycastResult then
		local normal = raycastResult.Normal
		local slopeAngle = math.deg(math.acos(normal.Y))
		local canStand = slopeAngle <= Config.Movement.slideThreshold
		State.cachedGroundInfo = {true, canStand, slopeAngle}
	else
		State.cachedGroundInfo = {false, false, 0}
	end

	State.lastGroundCheckTime = currentTime
	return unpack(State.cachedGroundInfo)
end

-- ===============================================
-- PHYSICS CALCULATIONS
-- ===============================================

local Physics = {}

function Physics.calculateSlideVelocity(horizontalVelocity, slopeAngle, dt)
	local slopeDir = Vector3.new(horizontalVelocity.X, 0, horizontalVelocity.Z).Unit

	if slopeDir.Magnitude == 0 then
		slopeDir = -State.lastCFrame.LookVector
	end

	local sinSlopeAngle = math.sin(math.rad(slopeAngle))
	local slopeAcceleration = Config.Movement.gravityForce * sinSlopeAngle * dt
	local slideVelocity = horizontalVelocity + slopeDir * slopeAcceleration

	slideVelocity = slideVelocity * (1 - 0.1 * dt)
	return slideVelocity
end

function Physics.calculateGroundMovement(moveDir, horizontalVelocity, dt)
	local moveMagnitude = moveDir.Magnitude

	if moveMagnitude > 0.1 then
		local worldMoveDir = State.lastCFrame:VectorToObjectSpace(
			Vector3.new(moveDir.X, 0, moveDir.Z)
		).Unit
		local targetVelocity = worldMoveDir * Config.Movement.baseSpeed * moveMagnitude

		local lerpAlpha = 1 - math.exp(-Config.Movement.acceleration * dt)
		return horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
	else
		local lerpAlpha = 1 - math.exp(-Config.Movement.deceleration * dt)
		return horizontalVelocity:Lerp(State.VECTOR_ZERO, lerpAlpha)
	end
end

function Physics.calculateAirMovement(moveDir, horizontalVelocity, dt)
	local moveMagnitude = moveDir.Magnitude

	if moveMagnitude > 0.1 then
		local worldMoveDir = State.lastCFrame:VectorToObjectSpace(
			Vector3.new(moveDir.X, 0, moveDir.Z)
		).Unit

		local airControlFactor = math.max(0, -State.currentVerticalVelocity / Config.Movement.airResistance)
		local targetVelocity = worldMoveDir * Config.Movement.baseSpeed * Config.Movement.airControl * airControlFactor

		local lerpAlpha = 1 - math.exp(-Config.Movement.acceleration * 0.3 * dt)
		return horizontalVelocity:Lerp(targetVelocity, lerpAlpha)
	else
		return horizontalVelocity * (1 - 20 * dt)
	end
end

function Physics.updateRotation(dt)
	local camCFrame = cam.CFrame
	local targetLook = Vector3.new(camCFrame.LookVector.X, 0, camCFrame.LookVector.Z).Unit
	local lerpAlpha = math.clamp(Config.Movement.turnSpeed * dt * 2, 0, 1)

	State.lastCFrame = State.lastCFrame:Lerp(
		CFrame.new(State.rigBody.Position, State.rigBody.Position + targetLook),
		lerpAlpha
	)
end

-- ===============================================
-- UI
-- ===============================================

local UI = {}

function UI.updateVelocityDisplay(horizontalVelocity)
	local currentTime = tick()

	if not State.vectorForcesUI or 
		currentTime - State.lastUiUpdateTime < Config.Performance.uiUpdateInterval then
		return
	end

	State.vectorForcesUI.X.Text = "X: " .. tostring(math.round(horizontalVelocity.X))
	State.vectorForcesUI.Y.Text = "Y: " .. tostring(math.round(State.currentVerticalVelocity))
	State.vectorForcesUI.Z.Text = "Z: " .. tostring(math.round(horizontalVelocity.Z))
	State.lastUiUpdateTime = currentTime
end

-- ===============================================
-- NETWORK OPTIMIZATION
-- ===============================================

local Network = {}

function Network.packData(velocity, cframe)
	local buffer = BitBuffer.Create()
	local config = Config.BitPacking

	buffer:WriteUnsigned(13, math.clamp(math.round(velocity.X + config.velocityOffsetX), 0, 6000))
	buffer:WriteUnsigned(13, math.clamp(math.round(velocity.Z + config.velocityOffsetZ), 0, 6000))
	buffer:WriteUnsigned(12, math.clamp(math.round(velocity.Y + config.velocityOffsetY), 0, 3000))

	local pos = cframe.Position
	buffer:WriteUnsigned(13, math.clamp(math.round((pos.X + config.positionOffset) * config.positionFactor), 0, 8191))
	buffer:WriteUnsigned(13, math.clamp(math.round((pos.Y + config.positionOffset) * config.positionFactor), 0, 8191))
	buffer:WriteUnsigned(13, math.clamp(math.round((pos.Z + config.positionOffset) * config.positionFactor), 0, 8191))

	local yaw = math.atan2(-cframe.RightVector.Z, cframe.RightVector.X)
	buffer:WriteUnsigned(9, math.clamp(math.round((yaw + math.pi) * config.yawScale), 0, 511))

	return buffer:ToBase64()
end

function Network.sendUpdate()
	local currentTime = tick()

	if currentTime - State.lastSentTime < Config.Network.sendInterval then
		return
	end

	local packedData = Network.packData(State.currentVelocity, State.lastCFrame)
	moveRigEvent:FireServer(packedData)

	State.lastSentVelocity = State.currentVelocity
	State.lastSentCFrame = State.lastCFrame
	State.lastSentTime = currentTime
end

-- ===============================================
-- MAIN PHYSICS UPDATE
-- ===============================================

local function updatePhysics(dt)
	local moveDir = State.humanoid.MoveDirection
	local grounded, canStand, slopeAngle = GroundDetection.checkGround()

	local justLanded = grounded and not State.isGrounded
	State.isGrounded = grounded
	State.isSliding = grounded and slopeAngle >= Config.Movement.slideThreshold

	local horizontalVelocity = Vector3.new(State.currentVelocity.X, 0, State.currentVelocity.Z)

	-- // Ground vs Air physics
	if grounded then
		-- // Handle landing
		if justLanded or State.currentVerticalVelocity < 0 then
			State.currentVerticalVelocity = 0
			State.canJump = true
		end

		-- // Ground movement
		if State.isSliding then
			horizontalVelocity = Physics.calculateSlideVelocity(horizontalVelocity, slopeAngle, dt)
		else
			horizontalVelocity = Physics.calculateGroundMovement(moveDir, horizontalVelocity, dt)
		end
	else
		-- // Air physics
		State.currentVerticalVelocity = math.max(
			State.currentVerticalVelocity - Config.Movement.gravityForce * dt,
			Config.Movement.terminalVelocity
		)
		horizontalVelocity = Physics.calculateAirMovement(moveDir, horizontalVelocity, dt)
	end

	State.currentVelocity = Vector3.new(
		math.clamp(horizontalVelocity.X, -Config.Movement.horizontalTerminalVelocity, Config.Movement.horizontalTerminalVelocity),
		State.currentVerticalVelocity,
		math.clamp(horizontalVelocity.Z, -Config.Movement.horizontalTerminalVelocity, Config.Movement.horizontalTerminalVelocity)
	)

	-- Update systems
	UI.updateVelocityDisplay(horizontalVelocity)
	Physics.updateRotation(dt)
end

-- ===============================================
-- INPUT HANDLING
-- ===============================================

local Input = {}

function Input.onJumpRequest()
	if State.isGrounded and State.canJump and not State.isSliding then
		State.currentVerticalVelocity = Config.Movement.jumpPower
		State.canJump = false
	end
end

function Input.onKeyPress(input, gameProcessed)
	if gameProcessed or input.UserInputType ~= Enum.UserInputType.Keyboard then
		return
	end

	if input.KeyCode == Enum.KeyCode.R then
		State.humanoid.Health = 0
	end
end

-- ===============================================
-- CHARACTER MANAGEMENT
-- ===============================================

local CharacterManager = {}

function CharacterManager.setupCharacterReferences(character)
	State.character = character
	State.humanoid = character:WaitForChild("Humanoid")
	State.humanoidRootPart = character:WaitForChild("HumanoidRootPart")

	local rigName = "Rig_" .. player.Name
	local rig = workspace:WaitForChild(rigName)
	State.rigHead = rig:WaitForChild("Head")
	State.rigBody = rig:WaitForChild("Body")

	State.playerGui = player:WaitForChild("PlayerGui")
	State.vectorForcesUI = State.playerGui:WaitForChild("Keys"):WaitForChild("VectorForces")

	cam.CameraSubject = State.rigHead

	Utils.updateRaycastFilter()
end

function CharacterManager.onCharacterAdded(character)
	Utils.cleanupConnections()
	CharacterManager.setupCharacterReferences(character)
	Utils.resetMovementState()

	Utils.addConnection(RunService.RenderStepped:Connect(function(dt)
		updatePhysics(dt)
		Network.sendUpdate()
	end))

	Utils.addConnection(UserInputService.JumpRequest:Connect(Input.onJumpRequest))
	Utils.addConnection(UserInputService.InputBegan:Connect(Input.onKeyPress))
end

function CharacterManager.onPlayerAdded(plr)
	if plr ~= player then return end

	Utils.addConnection(plr.CharacterAdded:Connect(CharacterManager.onCharacterAdded))

	if plr.Character then
		CharacterManager.onCharacterAdded(plr.Character)
	end
end

-- ===============================================
-- INITIALIZATION
-- ===============================================

local function initialize()
	Utils.addConnection(player.CharacterAdded:Connect(CharacterManager.onCharacterAdded))
	Utils.addConnection(Players.PlayerAdded:Connect(CharacterManager.onPlayerAdded))

	for _, plr in pairs(Players:GetPlayers()) do
		CharacterManager.onPlayerAdded(plr)
	end

	script.Destroying:Connect(Utils.cleanupConnections)

	if player.Character then
		CharacterManager.onCharacterAdded(player.Character)
	end
end

initialize()
