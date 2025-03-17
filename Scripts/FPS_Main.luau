--[[

TODO: Optomize Code

Add ammo system and reloading

--]]

-- // Variables

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local TweenService = game:GetService("TweenService")
local RunService = game:GetService("RunService")
local UserInputService = game:GetService("UserInputService")
local ContextActionService = game:GetService("ContextActionService")

local player = game.Players.LocalPlayer
local char = player.Character or player.CharacterAdded:Wait()
local hum = char:WaitForChild("Humanoid")

local leftArm = char:WaitForChild("Left Arm")
local rightArm = char:WaitForChild("Right Arm")

local cam = workspace.CurrentCamera
local NormalFieldOfView = cam.FieldOfView

local humanimator = hum.Animator

-- // ViewModels

local ViewModels = {
	Glock = ReplicatedStorage:WaitForChild("ViewModels"):WaitForChild("GlockViewModel"),
	Rifle = ReplicatedStorage:WaitForChild("ViewModels"):WaitForChild("RifleViewModel")
}

local ViewModel = ViewModels.Glock

local ViewModelOffset = CFrame.new(0, -0.5, 2) -- // Offset of ViewModel (Not for Swaying)

-- // UI

local VignetteUI = player:WaitForChild("PlayerGui"):WaitForChild("Vignette")

-- // Modules

local SpringModule = require(ReplicatedStorage:WaitForChild("Modules"):WaitForChild("Spring"))
local BulletEject = require(ReplicatedStorage:WaitForChild("Modules"):WaitForChild("BulletEjection"))
local FastCast = require(ReplicatedStorage:WaitForChild("Modules"):WaitForChild("FastCastRedux"))
local DMG = require(ReplicatedStorage:WaitForChild("Modules").DamageCounter)
local Shake = require(game.ReplicatedStorage:WaitForChild("Modules").CameraShaker)
local EaseAnimations = require(game.ReplicatedStorage:WaitForChild("Modules").Easimation)

-- // RemoteEvents

local HitEvent = ReplicatedStorage:WaitForChild("Remotes"):WaitForChild("Hit")
local ClientToServer = ReplicatedStorage:WaitForChild("Remotes").ClientToServer
local AttachGunEvent = ReplicatedStorage:WaitForChild("Remotes").AttachGunEvent
local BulletHitWall = ReplicatedStorage:WaitForChild("Remotes").BulletHitWall
local FireBulletEvent = ReplicatedStorage:WaitForChild("Remotes").FireBulletEvent
local StopSprint = ReplicatedStorage:WaitForChild("Remotes").StopSprint

-- // Weapon Stats

local Damage = nil
local CritDamage = nil
local MaxAmmo = nil
local AimPoint = nil
local FiringPoint = nil
local aimOffsetModule = nil
local EjectPoint = nil
local Bullet = nil
local GunModel = nil
local BulletCasing = nil
local Spread = nil
local Recoil = nil
local AccelerationModule = nil
local BulletSpeed = nil
local CharFiringPoint = nil

-- // Weapon States

local Idle = nil
local Walking = nil
local Running = nil
local Firing = nil
local Equip = nil

local CharIdle = nil
local CharWalking = nil 
local CharRunning = nil
local CharFiring = nil
local CharEquip  = nil

-- // Weapon State Flags

local db = false
local isAiming = false
local isEquipping = false

-- // Camera and Tweening

local lastCameraCF = cam.CFrame
local TweeninfoCamera = TweenInfo.new(0.4, Enum.EasingStyle.Quad, Enum.EasingDirection.Out)

-- // Sounds

local SlideBack = nil
local SlideForward = nil
local FireSound = nil
local EjectSound = nil
local ShootSound = nil
local AimSound = nil

-- // Sway-specific variables

local currentSwayOffset = CFrame.new()
local targetSwayOffset = CFrame.new()
local currentBobOffset = CFrame.new()
local dynamicLerpSpeed = 0.02
local swayIntensity = 2 -- // Controls the intensity of sway
local swaySpeed = 3 -- // Controls the speed of sway
local maxTiltPitch = math.rad(70)
local maxTiltYaw = math.rad(70) 

-- // Aim-specific variables

local currentAimCFrame = CFrame.new()
local targetAimCFrame = CFrame.new()
local FieldOfViewAim = 40

local LColor = char:WaitForChild("Body Colors").LeftArmColor3
local RColor = char:WaitForChild("Body Colors").RightArmColor3

-- // Helper function to create RaycastParams

local function CreateRaycastParams(filterDescendants)
	local params = RaycastParams.new()
	params.FilterType = Enum.RaycastFilterType.Exclude
	params.IgnoreWater = true
	params.FilterDescendantsInstances = filterDescendants
	return params
end

-- // Helper function to create FastCast behavior

local function CreateFastCastBehavior(raycastParams, cosmeticBulletContainer)
	local behavior = FastCast.newBehavior()
	behavior.RaycastParams = raycastParams
	behavior.AutoIgnoreContainer = false
	behavior.HighFidelitySegmentSize = 1
	behavior.MaxDistance = 1000
	behavior.CosmeticBulletContainer = cosmeticBulletContainer
	return behavior
end

-- // Client setup
local clientCastParams = CreateRaycastParams({char, workspace.Visuals})
local caster = FastCast.new()
local casterBehavior = CreateFastCastBehavior(clientCastParams, workspace.Visuals)

-- // Server setup
local serverCastParams = CreateRaycastParams(workspace.Visuals:GetDescendants())
local casterServer = FastCast.new()
local casterBehaviorServer = CreateFastCastBehavior(serverCastParams, workspace.Visuals)

local camShake = Shake.new(Enum.RenderPriority.Camera.Value, function(shakeCf)
	cam.CFrame = cam.CFrame * shakeCf
end)

camShake:Start()

local TTAnims = TweenInfo.new(0.4, Enum.EasingStyle.Quad, Enum.EasingDirection.Out)

local function SetVariables()
	local StatsModule = require(NewViewModel:WaitForChild("Settings"))
	local WeaponStats = StatsModule.WeaponStats
	local WeaponReferences = StatsModule.WeaponReferences
	local PlayerAnimations = StatsModule.PlayerAnimations
	local Animations = StatsModule.Animations

	local animator = NewViewModel:WaitForChild("Humanoid").Animator

	-- // Sets Stats of Guns
	Damage = WeaponStats.Damage
	CritDamage = WeaponStats.CritDamage
	MaxAmmo = WeaponStats.MaxAmmo
	Spread = WeaponStats.Spread
	Recoil = WeaponStats.Recoil
	FireDelay = WeaponStats.FireDelay
	aimOffsetModule = WeaponStats.AimOffset
	AccelerationModule = WeaponStats.Acceleration
	BulletSpeed = WeaponStats.BulletSpeed

	-- // Sets References of Guns
	AimPoint = WeaponReferences.AimPoint
	FiringPoint = WeaponReferences.FirePoint
	EjectPoint = WeaponReferences.EjectPoint
	Bullet = WeaponReferences.Bullet
	BulletCasing = WeaponReferences.BulletCasing
	GunModel = WeaponReferences.GunModel
	CharFiringPoint = WeaponReferences.GunModelFiringPoint

	-- // Set Animations
	CharIdle = humanimator:LoadAnimation(PlayerAnimations.Idle)
	CharWalking = humanimator:LoadAnimation(PlayerAnimations.Walking)
	CharRunning = humanimator:LoadAnimation(PlayerAnimations.Running)
	CharFiring = humanimator:LoadAnimation(PlayerAnimations.Firing)
	CharEquip = humanimator:LoadAnimation(PlayerAnimations.Equip)

	Idle = animator:LoadAnimation(Animations.Idle)
	Walking = animator:LoadAnimation(Animations.Walking)
	Running = animator:LoadAnimation(Animations.Running)
	Firing = animator:LoadAnimation(Animations.Firing)
	Equip = animator:LoadAnimation(Animations.Equip)

	-- // Set Priorities using a table
	local animationPriorities = {
		[Equip] = Enum.AnimationPriority.Action2,
		[Idle] = Enum.AnimationPriority.Action,
		[Walking] = Enum.AnimationPriority.Action,
		[Running] = Enum.AnimationPriority.Action3,
		[Firing] = Enum.AnimationPriority.Action4,
		[CharEquip] = Enum.AnimationPriority.Action3,
		[CharIdle] = Enum.AnimationPriority.Action,
		[CharWalking] = Enum.AnimationPriority.Action,
		[CharRunning] = Enum.AnimationPriority.Action2,
		[CharFiring] = Enum.AnimationPriority.Action4,
	}

	for animation, priority in pairs(animationPriorities) do
		animation.Priority = priority
	end

	-- // Fastcast and Others
	local NewBullet = Bullet:Clone()
	casterBehavior.Acceleration = AccelerationModule
	casterBehaviorServer.Acceleration = AccelerationModule
	casterBehavior.CosmeticBulletTemplate = NewBullet
	casterBehaviorServer.CosmeticBulletTemplate = NewBullet

	local filterInstances = {char, workspace.Visuals, NewViewModel:GetDescendants()}
	clientCastParams.FilterDescendantsInstances = filterInstances
	casterBehavior.FilterDescendantsInstances = filterInstances

	-- // Cache Sounds
	local GunSounds = GunModel:WaitForChild("Sounds")
	SlideBack = GunSounds.BBackward
	SlideForward = GunSounds.BForward
	FireSound = GunSounds.Shoot
	EjectSound = GunSounds.BulletEject
	AimSound = GunSounds.Aim
end

local function AttachGunModel()
	AttachGunEvent:FireServer(GunModel)
end

local function SwitchViewModel(newWeapon)
	if not ViewModels[newWeapon] then return end  
	if NewViewModel then NewViewModel:Destroy() end

	NewViewModel = ViewModels[newWeapon]:Clone()
	NewViewModel.Parent = cam
	NewViewModel.PrimaryPart = NewViewModel:WaitForChild("HumanoidRootPart")

	local leftArm = NewViewModel:WaitForChild("LeftArm")
	local rightArm = NewViewModel:WaitForChild("RightArm")

	leftArm.Color = LColor
	rightArm.Color = RColor

	SetVariables()

	AttachGunModel()

	EquipWeapon()

end

local currentWeapon = "Glock"

local weaponKeys = {
	[Enum.KeyCode.One] = "Glock",
	[Enum.KeyCode.Two] = "Rifle"
}

UserInputService.InputBegan:Connect(function(input, gameProcessed)
	if gameProcessed then return end

	local newWeapon = weaponKeys[input.KeyCode]
	if newWeapon and newWeapon ~= currentWeapon then
		currentWeapon = newWeapon
		SwitchViewModel(newWeapon)
	end
end)

-- // Preload Animations

local function PreloadAnimations(animator, animationTable)
	for _, anim in pairs(animationTable) do
		animator:LoadAnimation(anim)
	end
end

PreloadAnimations(humanimator, {
	CharIdle, CharWalking, CharRunning, CharFiring, CharEquip
})

local function PreloadWeaponAnimations(viewModel)
	local weaponStats = require(viewModel.Settings)
	PreloadAnimations(viewModel:FindFirstChild("Humanoid").Animator, {
		weaponStats.Animations.Idle,
		weaponStats.Animations.Walking,
		weaponStats.Animations.Running,
		weaponStats.Animations.Firing,
		weaponStats.Animations.Equip
	})
end

for _, viewModel in pairs(ViewModels) do
	PreloadWeaponAnimations(viewModel)
end

-- // ViewModel Functionality

function PlayAnimationsViewModel()
	-- // Early exit if conditions are not met
	
	if not NewViewModel or not char or not hum or isEquipping then
		return
	end

	-- // Calculate movement state
	local speed = hum.MoveDirection.Magnitude
	local isMoving = speed > 0
	local isRunning = isMoving and hum.WalkSpeed > 15

	-- // Stop all animations if not moving
	
	if not isMoving then
		if Idle and not Idle.IsPlaying then
			Idle:Play()
			CharIdle:Play()
		end
		if Walking and Walking.IsPlaying then
			Walking:Stop()
		end
		if Running and Running.IsPlaying then
			Running:Stop()
			CharRunning:Stop(0.2)
		end
		return
	end

	-- // Handle running state
	
	if isRunning then
		if Idle and Idle.IsPlaying then
			Idle:Stop()
		end
		if Walking and Walking.IsPlaying then
			Walking:Stop()
		end
		if Running and not Running.IsPlaying then
			Running:Play()
			CharRunning:Play(0.2)
		end
		return
	end

	-- // Handle walking state
	
	if Idle and Idle.IsPlaying then
		Idle:Stop()
	end
	if Walking and not Walking.IsPlaying then
		Walking:Play()
		CharWalking:Play(0.2)
	end
	if Running and Running.IsPlaying then
		Running:Stop()
		CharRunning:Stop(0.2)
	end
end

local currentSwayOffset = CFrame.new()
local delayedSwayOffset = CFrame.new()

local function SwayViewmodel(dt)
	if not NewViewModel or not hum then return CFrame.new() end

	local cameraCF = cam.CFrame
	local lastPitch, lastYaw = select(1, lastCameraCF:ToEulerAnglesYXZ())
	local currentPitch, currentYaw = select(1, cameraCF:ToEulerAnglesYXZ())

	local tiltAmountPitch = math.clamp((currentPitch - lastPitch) * swayIntensity, -maxTiltPitch, maxTiltPitch)
	local tiltAmountYaw = math.clamp((currentYaw - lastYaw) * swayIntensity, -maxTiltYaw, maxTiltYaw)

	local targetSwayOffset = CFrame.Angles(tiltAmountPitch * 0.5, 0, -tiltAmountYaw)

	if not isAiming then
		targetSwayOffset *= CFrame.new(math.sin(tick() * 5) * 0.04, math.cos(tick() * 3) * 0.04, 0)
	end

	local alpha = math.min(dt * 10, 1)
	delayedSwayOffset, currentSwayOffset = delayedSwayOffset:Lerp(targetSwayOffset, alpha), currentSwayOffset:Lerp(delayedSwayOffset, alpha)

	lastCameraCF = cameraCF
	return currentSwayOffset
end

local function ViewModelBobbing()
	if not NewViewModel or not hum then return CFrame.new() end

	local speed = hum.MoveDirection.Magnitude

	local bobSpeed = 6 * (hum.WalkSpeed / 10)
	local bobAmount = 0.1 * (hum.WalkSpeed / 10)
	local tiltAmount = 0.1 * (hum.WalkSpeed / 10)

	local time = tick() * bobSpeed

	local bobX = math.sin(time) * bobAmount
	local bobY = math.sin(time * 2) * bobAmount

	local tiltAngle = math.sin(time) * tiltAmount

	local targetBobOffset = CFrame.new(bobX, bobY, 0) * CFrame.Angles(0, 0, tiltAngle)

	if speed > 0 then
		currentBobOffset = currentBobOffset:Lerp(targetBobOffset, 0.1)
	else
		currentBobOffset = currentBobOffset:Lerp(CFrame.new(), 0.1)
	end

	return currentBobOffset
end

function EquipWeapon()
	if Equip then
		isEquipping = true
		Equip:Play()
		CharEquip:Play()
		Equip:GetMarkerReachedSignal("SlideBack"):Connect(function()
			SlideBack:Play()
		end)
		Equip:GetMarkerReachedSignal("SlideForward"):Connect(function()
			SlideForward:Play()
		end)
		Equip.Stopped:Wait() 
		isEquipping = false 
	end
end

-- // Aiming Functions

local function LerpCFrame(currentCFrame, targetCFrame, alpha)
	return currentCFrame:Lerp(targetCFrame, alpha)
end

local function AimFunction()
	if not NewViewModel.PrimaryPart or not AimPoint then return CFrame.new() end

	if isAiming then
		local currentCFrame = NewViewModel.PrimaryPart.CFrame
		local cameraLookVector = cam.CFrame.LookVector
		local attachmentOffset = AimPoint.WorldPosition - NewViewModel.PrimaryPart.Position
		local aimOffset = CFrame.new(attachmentOffset):Inverse()
		targetAimCFrame = CFrame.new(0, 0, attachmentOffset.Magnitude / 2) * aimOffsetModule
		TweenService:Create(cam, TweeninfoCamera, {FieldOfView = FieldOfViewAim}):Play()
		TweenService:Create(VignetteUI.Holder, TweeninfoCamera, {GroupTransparency = 0.6}):Play()
	else
		targetAimCFrame = CFrame.new()
		TweenService:Create(cam, TweeninfoCamera, {FieldOfView = NormalFieldOfView}):Play()
		TweenService:Create(VignetteUI.Holder, TweeninfoCamera, {GroupTransparency = 1}):Play()
	end

	currentAimCFrame = LerpCFrame(currentAimCFrame, targetAimCFrame, 0.2)

	return currentAimCFrame
end

UserInputService.InputBegan:Connect(function(input, gameProcessed)
	if gameProcessed then return end
	if input.UserInputType == Enum.UserInputType.MouseButton2 and not isEquipping and NewViewModel ~= nil then
		isAiming = true
		GunModel:WaitForChild("Sounds"):WaitForChild("Aim"):Play()
		script.isAiming.Value = isAiming
		StopSprint:Fire()
	end
end)

UserInputService.InputEnded:Connect(function(input, gameProcessed)
	if gameProcessed then return end
	if input.UserInputType == Enum.UserInputType.MouseButton2 then
		isAiming = false
		script.isAiming.Value = isAiming
	end
end)

-- // Attach ViewModels to Camera

function BindToCamera(deltaTime)
	if not NewViewModel then
		NewViewModel = ViewModel:Clone()
		NewViewModel.Parent = cam
		NewViewModel.PrimaryPart = NewViewModel:WaitForChild("HumanoidRootPart")

		NewViewModel:WaitForChild("LeftArm").Color = LColor
		NewViewModel:WaitForChild("RightArm").Color = RColor

		SetVariables()

		AttachGunModel()

		EquipWeapon()
	end

	local BobbingOffset = ViewModelBobbing()
	local SwayOffset = SwayViewmodel(deltaTime)
	local AimOffset = AimFunction()

	PlayAnimationsViewModel()
	
	local finalCFrame = cam.CFrame * ViewModelOffset * BobbingOffset * SwayOffset * AimOffset

	if NewViewModel.PrimaryPart then
		NewViewModel:SetPrimaryPartCFrame(finalCFrame)
	end
end

-- // Firing Function

local function GetSpreadDirection(direction, spreadAngle)
	local spreadDirection = direction + (Vector3.new(
		math.random() - 0.5,
		math.random() - 0.5,
		math.random() - 0.5
		) * math.tan(spreadAngle))
	return spreadDirection.Unit
end

local function OnHit(cast, result, velocity, bullet)
	local hit = result.Instance
	local characterHit = hit:FindFirstAncestorOfClass("Model")
	if characterHit and characterHit:FindFirstChild("Humanoid") then
		DMG.Highlight(hit)
		game.ReplicatedStorage:WaitForChild("Sounds").HitPlayer:Play()
		HitEvent:FireServer(Damage, CritDamage, hit, characterHit.Humanoid, bullet)
	else
		BulletHitWall:FireServer(bullet.Position, hit)
	end
end

BulletHitWall.OnClientEvent:Connect(function(bulletPart, hit)
	local HitParticles = game.ReplicatedStorage:WaitForChild("Particles").HitParticles:Clone()
	HitParticles.Parent = workspace.Visuals
	HitParticles.Position = bulletPart
	local Sound = HitParticles:WaitForChild("Hit")
	Sound:Play()
	for _, v in ipairs(HitParticles:GetChildren()) do
		if v:IsA("ParticleEmitter") and hit then 
			v.Color = ColorSequence.new(hit.Color)
			v:Emit(20)
		end
	end
	task.wait(1)
	HitParticles:Destroy()
end)

local function OnTerminated(cast, bullet)
	if bullet and bullet:IsDescendantOf(workspace.Visuals) then
		bullet:Destroy()
	end
end

local function FiringFunction()
	if isEquipping == true or NewViewModel == nil then return end

	Firing:Play()
	CharFiring:Play()

	local NewRecoil = Recoil

	if isAiming then
		NewRecoil /= 2
	else
		NewRecoil = Recoil
	end
	camShake:ShakeOnce(NewRecoil, 4, 0, 0.4)
	BulletEject.BulletEject(BulletCasing:Clone(), 30, EjectPoint, 70)

	for _, v in pairs(FiringPoint:GetDescendants()) do
		if v:IsA("ParticleEmitter") then v:Emit(10) end
	end

	for _, v in pairs(rightArm:GetDescendants()) do
		if v:IsA("ParticleEmitter") then ClientToServer:FireServer(v) end
	end

	local origin = FiringPoint.WorldPosition
	local spreadAngle = isAiming and Spread / 5 or Spread
	local spreadDirection = GetSpreadDirection(cam.CFrame.LookVector, spreadAngle)

	caster:Fire(origin, spreadDirection.Unit, BulletSpeed, casterBehavior)
	
	FireBulletEvent:FireServer(origin, spreadDirection.Unit, FireSound)
end

FireBulletEvent.OnClientEvent:Connect(function(client, origin, endpos)
	if client ~= game.Players.LocalPlayer then
		casterServer:Fire(origin, endpos, BulletSpeed, casterBehaviorServer)
	end
end)

local function HandleLengthChanged(cast, lastPoint, direction, length, velocity, bullet)
	if bullet then
		local bulletLength = bullet.Size.Z / 2
		local offset = length - bulletLength
		bullet.CFrame = CFrame.lookAt(lastPoint, lastPoint + direction) * CFrame.Angles(0, math.rad(-90), 0)
	end
end

local function HandleCastTerminating(cast)
	local bullet = cast.RayInfo.CosmeticBulletObject
	OnTerminated(cast, bullet)
end

caster.LengthChanged:Connect(HandleLengthChanged)
caster.RayHit:Connect(OnHit)
caster.CastTerminating:Connect(HandleCastTerminating)

casterServer.LengthChanged:Connect(HandleLengthChanged)
casterServer.CastTerminating:Connect(HandleCastTerminating)
casterServer.RayHit:Connect(function(cast, result, velocity, bullet)
	HandleCastTerminating(cast)
end)

-- // Input Handling

local canFire = true

UserInputService.InputBegan:Connect(function(input, gameProcessed)
	if gameProcessed then return end
	if input.UserInputType == Enum.UserInputType.MouseButton1 and canFire then
		isFiring = true
		canFire = false 
		while isFiring do
			FiringFunction()
			task.wait(FireDelay)
		end
		canFire = true
	end
end)

UserInputService.InputEnded:Connect(function(input)
	if input.UserInputType == Enum.UserInputType.MouseButton1 then
		isFiring = false
	end
end)

RunService.RenderStepped:Connect(BindToCamera)

hum.Died:Connect(function()
	cam:ClearAllChildren()
	TweenService:Create(cam, TweeninfoCamera, {FieldOfView = NormalFieldOfView}):Play()
	TweenService:Create(VignetteUI.Holder, TweeninfoCamera, {GroupTransparency = 1}):Play()
	isAiming = false
	script.isAiming.Value = isAiming
	script:Destroy()
end)
