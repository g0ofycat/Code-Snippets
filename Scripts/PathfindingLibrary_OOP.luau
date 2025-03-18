--[[Client:
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local UserInputService = game:GetService("UserInputService")
local PathfindingModule = require(ReplicatedStorage:WaitForChild("PathfindingModule"))

local pathfinder = PathfindingModule.new()

pathfinder:SetVisualization(true, { -- // Set to true to visualize grid, if youre making massive grids then don't set this to true
	pointSize = 1, -- // Size of the Visual Part
	lineSize = 0.2, -- // Size of the Grid Part
	pointColor = Color3.fromRGB(102, 0, 255), -- // Color of the Visual Part
	lineColor = Color3.fromRGB(136, 0, 255) -- // Color of the Grid Part
})

local player = Players.LocalPlayer
local character = player.Character or player.CharacterAdded:Wait()

local grid = pathfinder:CreateWorldGrid(Vector3.new(0, 0, 0), Vector3.new(100, 1000, 100), 2, true) -- // Center: Center of the grid, Size: Size of the grid. Resoulution: Distance between each point (Recommended: 2 since it needs to detect collision). HeightCheck: Raycasts each point to each for the hit parts Y, more accurate collisions

UserInputService.InputBegan:Connect(function(input, gameProcessed)
	if gameProcessed then return end

	if input.UserInputType == Enum.UserInputType.MouseButton1 then
		local mouse = player:GetMouse()
		local raycastResult = workspace:Raycast(mouse.UnitRay.Origin, mouse.UnitRay.Direction * 10000)

		if raycastResult then
			local startPosition = character.HumanoidRootPart.Position
			local targetPosition = raycastResult.Position

			-- // Find a path using the module
			
			local path, algorithm = pathfinder:FindBestPath(
				character.PrimaryPart.Position, -- // Start position of the NPC (Vector3)
				targetPosition, -- // End position of the NPC (Vector3)
				grid, -- // Grid if you want to use the other pathfinding algorithms other than Roblox
				{"Dijkstra", "Roblox"}, -- // Possible Names: Roblox (PathfindingService), AStar (A* Pathfinding Algorithm), Dijkstra (Dijkstra Pathfinding Algorithm), Grid (AStar without the diagonal vecotrs, moves in a grid-like pattern)
				-- // The table above can store multiple pathfinding system as a fallback, if A* Fails then the normal Roblox one can be used. There can be as much as you want.
				true -- // Set to true to visualize path
			)

			if path then
				print("Found path using", algorithm, "with", #path, "waypoints")
				
				-- // Function to actually follow the path, uses your character as an example
				
				pathfinder:FollowPath(character, path)
			else
				print("No path found!")
			end
		end
	end
end)
--]]

---------------------------------------------
-- // Variables
---------------------------------------------
local PathfindingService = game:GetService("PathfindingService")
local RunService = game:GetService("RunService")

local PathfindingModule = {}
PathfindingModule.__index = PathfindingModule

---------------------------------------------
-- // Functionality
---------------------------------------------
function PathfindingModule.new()
	local self = setmetatable({}, PathfindingModule)
	self.currentAlgorithm = nil -- // Debugging Variable
	self.resolution = 1

	self.algorithms = {
		Roblox = function(start, goal) 
			self.currentAlgorithm = "Roblox"
			return self:RobloxPathfinding(start, goal) 
		end,
		AStar = function(start, goal, grid) 
			self.currentAlgorithm = "AStar"
			return self:AStarPathfinding(start, goal, grid) 
		end,
		Dijkstra = function(start, goal, grid) 
			self.currentAlgorithm = "Dijkstra"
			return self:DijkstraPathfinding(start, goal, grid) 
		end,
		Grid = function(start, goal, grid) 
			self.currentAlgorithm = "Grid"
			return self:AStarPathfinding(start, goal, grid) 
		end
	}

	-- // Visualization properties
	self.visualFolder = nil
	self.visualization = {
		enabled = false,
		pointSize = 0.5,
		lineSize = 0.2,
		pointColor = Color3.fromRGB(0, 162, 255),
		lineColor = Color3.fromRGB(255, 255, 0)
	}

	return self
end

-- // Find a path using the best available algorithm

function PathfindingModule:FindBestPath(start: Vector3, goal: Vector3, grid: Instance, algorithmNames: Table, visualize: boolean)
	for _, name in ipairs(algorithmNames) do
		local path = self.algorithms[name] and self.algorithms[name](start, goal, grid)
		if path then 
			if visualize then -- // We dont check for the self visualization as those are 2 seperate visualizations
				self:VisualizePath(path)
			end

			return path, name 
		end
	end
	return nil, "No valid path found"
end

---------------------------------------------
-- // Helper Functions
---------------------------------------------

local function Vector3toString(vector)
	return vector.X .. "," .. vector.Y .. "," .. vector.Z
end

function PathfindingModule:GetNeighbors(position: Vector3, grid: table, resolution: number)
	resolution = resolution or self.resolution
	local neighbors = {}

	-- // Base movement (4-directional for grid-based movement)
	local offsets = {
		Vector3.new(resolution, 0, 0),   -- // Right
		Vector3.new(-resolution, 0, 0),  -- // Left
		Vector3.new(0, 0, resolution),   -- // Forward
		Vector3.new(0, 0, -resolution),  -- // Backward
		Vector3.new(0, resolution, 0),   -- // Up
		Vector3.new(0, -resolution, 0),  -- // Down
	}

	-- // If NOT using the Grid algorithm, allow diagonal movement
	if self.currentAlgorithm ~= "Grid" then
		local diagonalOffsets = {
			Vector3.new(resolution, 0, resolution),
			Vector3.new(resolution, 0, -resolution),
			Vector3.new(-resolution, 0, resolution),
			Vector3.new(-resolution, 0, -resolution),
		}
		for _, offset in ipairs(diagonalOffsets) do
			table.insert(offsets, offset)
		end
	end

	local raycastParams = RaycastParams.new()
	raycastParams.FilterType = Enum.RaycastFilterType.Exclude
	raycastParams.FilterDescendantsInstances = {}
	raycastParams.IgnoreWater = true

	-- // Process all valid neighbor positions
	for _, offset in ipairs(offsets) do
		local neighborPos = position + offset
		local neighborKey = Vector3toString(neighborPos)

		-- // Check if the position exists in the grid
		if grid[neighborKey] then
			local direction = neighborPos - position
			local rayOrigin = position + Vector3.new(0, 1, 0)
			local rayResult = workspace:Raycast(rayOrigin, direction, raycastParams)

			-- // If no obstacle is hit (rayResult is nil), the neighbor is valid
			if not rayResult then
				table.insert(neighbors, grid[neighborKey])
			end
		end
	end

	return neighbors
end

local function heuristic(a, b) -- // Euclidean-distance Heuristic
	return (b - a).Magnitude
end

local function stringToVector3(str)
	local x, y, z = str:match("([^,]+),([^,]+),([^,]+)")
	return Vector3.new(tonumber(x), tonumber(y), tonumber(z))
end

local function newBinaryHeap() -- // I'd recommend watching videos about pathfinding and heuristics: https://www.youtube.com/watch?v=VEYSSANa-cw
	local heap = {size = 0}
	local data = {}

	local function swap(i, j) -- // Swap the 2 arguments
		data[i], data[j] = data[j], data[i]
	end

	local function bubbleUp(index) -- // Move an element up (For min heap)
		while index > 1 do
			local parent = math.floor(index / 2)
			if data[index].fScore < data[parent].fScore then
				swap(index, parent)
				index = parent
			else
				break
			end
		end
	end

	local function bubbleDown(index) -- // Move an element down (For min heap)
		while true do
			local smallest = index
			local left = index * 2
			local right = index * 2 + 1

			if left <= heap.size and data[left].fScore < data[smallest].fScore then
				smallest = left
			end

			if right <= heap.size and data[right].fScore < data[smallest].fScore then
				smallest = right
			end

			if smallest == index then
				break
			end

			swap(index, smallest)
			index = smallest
		end
	end

	function heap:push(node, fScore) -- // To insert a new element
		heap.size = heap.size + 1
		data[heap.size] = {node = node, fScore = fScore}
		bubbleUp(heap.size)
	end

	function heap:pop() -- // Remove and return smallest 'fScore' from the heap
		if heap.size == 0 then
			return nil
		end

		local min = data[1]
		data[1] = data[heap.size]
		heap.size = heap.size - 1

		if heap.size > 0 then
			bubbleDown(1)
		end

		return min.node, min.fScore
	end

	function heap:isEmpty()
		return heap.size == 0
	end

	return heap
end

---------------------------------------------
-- // Roblox Pathfinder (Mainly use this as a last resort in the fallback table or you just want normal pathfinding)
---------------------------------------------

function PathfindingModule:RobloxPathfinding(start, goal) -- // I don't really think this needs documentation
	local path = PathfindingService:CreatePath({
		AgentRadius = 2,
		AgentHeight = 5,
		WaypointSpacing = 2,
		AgentCanJump = true
	})

	local success, errorMessage = pcall(function()
		path:ComputeAsync(start, goal)
	end)

	if success and path.Status == Enum.PathStatus.Success then
		return path:GetWaypoints()
	else
		warn("Roblox pathfinding failed: " .. (errorMessage or path.Status.Name))
		return nil
	end
end

---------------------------------------------
-- // A* Pathfinder (Best for finding Optimal Paths (More Costly), Not Recommened for 3D as it's commonly used in things like graphs and other 2D Spaces)
---------------------------------------------
function PathfindingModule:AStarPathfinding(start, goal, grid)
	if not grid or not next(grid) then
		if self.currentAlgorithm == "Grid" then
			warn("Grid pathfinding requires a valid grid")
		else
			warn("AStar pathfinding requires a valid grid")
		end
		return nil
	end

	local startKey = Vector3toString(start)
	local goalKey = Vector3toString(goal)

	-- // Try to snap start and goal to nearest grid positions if they're not exactly on grid points
	
	if not grid[startKey] then -- // Snaps to closest point (Start)
		local closestDist = math.huge
		local closestPos = nil
		for key, pos in pairs(grid) do
			local dist = (pos - start).Magnitude
			if dist < closestDist then
				closestDist = dist
				closestPos = pos
			end
		end
		if closestPos and closestDist < 5 then
			start = closestPos
			startKey = Vector3toString(start)
		else
			warn("Start position not in grid and no nearby point found: ", startKey)
			return nil
		end
	end

	if not grid[goalKey] then -- // Snaps to closest point (Goal)
		local closestDist = math.huge
		local closestPos = nil
		for key, pos in pairs(grid) do
			local dist = (pos - goal).Magnitude
			if dist < closestDist then
				closestDist = dist
				closestPos = pos
			end
		end
		if closestPos and closestDist < 5 then
			goal = closestPos
			goalKey = Vector3toString(goal)
		else
			warn("Goal position not in grid and no nearby point found: ", goalKey)
			return nil
		end
	end

	start = grid[startKey]
	goal = grid[goalKey]

	local openSet = newBinaryHeap()
	openSet:push(start, 0)

	local cameFrom = {} -- // Stores the previous node in the best path
	local gScore = {}  -- // Cost from start to each node
	local fScore = {}  -- // Estimated total cost (gScore + heuristic) to reach the goal

	gScore[startKey] = 0
	fScore[startKey] = heuristic(start, goal)

	-- // Recommened to read the newBinaryHeap() comments to know whats actually happening
	
	while not openSet:isEmpty() do
		local current = openSet:pop() -- // Node with lowest fScore
		local currentKey = Vector3toString(current) -- // Converted to string

		if current == goal or currentKey == goalKey then -- // See if goal reached
			local path = {}
			local curr = current
			while curr do
				table.insert(path, 1, curr) -- // Add node(s)
				local currKey = Vector3toString(curr)
				curr = cameFrom[currKey] -- // Move to previous node(s)
			end
			return path
		end
		
		-- // If this path to the neighbor is better than any previous one
		for _, neighbor in ipairs(self:GetNeighbors(current, grid) or {}) do
			local neighborKey = Vector3toString(neighbor)
			local tentativeGScore = (gScore[currentKey] or math.huge) + (neighbor - current).Magnitude

			if tentativeGScore < (gScore[neighborKey] or math.huge) then
				cameFrom[neighborKey] = current -- // Update best previous node
				gScore[neighborKey] = tentativeGScore -- // Update cost from start
				fScore[neighborKey] = tentativeGScore + heuristic(neighbor, goal)
				openSet:push(neighbor, fScore[neighborKey])
			end
		end
	end

	if self.currentAlgorithm == "Grid" then
		warn("Grid pathfinding could not find a path")
	else
		warn("AStar pathfinding could not find a path")
	end
	return nil
end

function PathfindingModule:CreateWorldGrid(center: Vector3, size: Vector3, resolution: number, heightCheck: boolean?, bufferDistance: number?)
	heightCheck = heightCheck ~= false
	bufferDistance = bufferDistance or 2
	self.resolution = resolution

	local grid = {}
	local halfSize = size * 0.5
	local minPos = center - halfSize
	local maxPos = center + halfSize

	local raycastParams = RaycastParams.new()
	raycastParams.IgnoreWater = true
	raycastParams.FilterType = Enum.RaycastFilterType.Include

	local rayStartHeight = maxPos.Y + 50
	local downVector = Vector3.new(0, -size.Y - 100, 0)
	local upOffset = Vector3.new(0, 1, 0)
	local gridPoints = 0

	local directions = {
		Vector3.new(bufferDistance, 0, 0),
		Vector3.new(-bufferDistance, 0, 0),
		Vector3.new(0, 0, bufferDistance),
		Vector3.new(0, 0, -bufferDistance),
	}

	local overlapParams = OverlapParams.new()
	overlapParams.FilterType = Enum.RaycastFilterType.Exclude
	local partsInRegion = workspace:GetPartBoundsInBox(CFrame.new(center), size, overlapParams)
	raycastParams.FilterDescendantsInstances = partsInRegion

	-- // Batch process grid points 
	for x = minPos.X, maxPos.X, resolution do
		local roundedX = math.round(x / resolution) * resolution
		for z = minPos.Z, maxPos.Z, resolution do
			local roundedZ = math.round(z / resolution) * resolution
			local pos

			if heightCheck then
				local top = Vector3.new(roundedX, rayStartHeight, roundedZ)
				local result = workspace:Raycast(top, downVector, raycastParams)

				if not result then continue end
				pos = Vector3.new(roundedX, result.Position.Y, roundedZ)
			else
				pos = Vector3.new(roundedX, minPos.Y, roundedZ)
			end

			local rayOrigin = pos + upOffset
			local isTooCloseToWall = false

			for _, direction in directions do -- // Using cached directions
				local rayResult = workspace:Raycast(rayOrigin, direction, raycastParams)
				if rayResult and (rayResult.Position - rayOrigin).Magnitude < bufferDistance then
					isTooCloseToWall = true
					break
				end
			end

			if not isTooCloseToWall then
				grid[Vector3toString(pos)] = pos
				gridPoints = gridPoints + (heightCheck and 1 or 0.5)
			end
		end
	end

	print("Created world grid with " .. gridPoints .. " points")

	if self.visualization.enabled then -- // Visualze if visualization is enabled
		self:VisualizeGrid(grid)
	end

	return grid
end

---------------------------------------------
-- // Dijkstra Pathfinder (Used for mainly finding the shortest path between 2 nodes, more costly)
---------------------------------------------
function PathfindingModule:DijkstraPathfinding(start, goal, grid)
	if not grid or not next(grid) then
		warn("Dijkstra pathfinding requires a valid grid")
		return nil
	end

	local startKey = Vector3toString(start) -- // Turn the start and end positions into a string
	local goalKey = Vector3toString(goal) -- // Turn the goal and end positions into a string

	-- // Snap start to nearest grid point if not found
	if not grid[startKey] then
		local closestDist = math.huge
		local closestPos = nil
		for key, pos in pairs(grid) do
			local dist = (pos - start).Magnitude
			if dist < closestDist then
				closestDist = dist
				closestPos = pos
			end
		end
		if closestPos and closestDist < 5 then
			start = closestPos
			startKey = Vector3toString(start)
		else
			warn("Start position not found in grid and no nearby point found: ", startKey)
			return nil
		end
	end

	-- // Snap goal to nearest grid point if not found
	if not grid[goalKey] then
		local closestDist = math.huge
		local closestPos = nil
		for key, pos in pairs(grid) do
			local dist = (pos - goal).Magnitude
			if dist < closestDist then
				closestDist = dist
				closestPos = pos
			end
		end
		if closestPos and closestDist < 5 then
			goal = closestPos
			goalKey = Vector3toString(goal)
		else
			warn("Goal position not found in grid and no nearby point found: ", goalKey)
			return nil
		end
	end

	local distances = {} -- // For tracking shortest distance from start to every other position
	local previous = {} -- // For reconstructing shortest path
	local openSet = newBinaryHeap()
	local visited = {} -- // Track processed nodes

	for posKey, pos in pairs(grid) do
		distances[posKey] = math.huge
		previous[posKey] = nil
	end

	distances[startKey] = 0
	openSet:push(grid[startKey], 0) -- // Push the actual position from grid

	while not openSet:isEmpty() do
		local current, currentDistance = openSet:pop()
		local currentKey = Vector3toString(current)

		if visited[currentKey] then
			continue
		end
		visited[currentKey] = true

		-- // Check if goal is reached using keys
		if currentKey == goalKey then
			break
		end

		-- // Check each neighbor
		for _, neighbor in ipairs(self:GetNeighbors(current, grid)) do
			local neighborKey = Vector3toString(neighbor)
			if not visited[neighborKey] and grid[neighborKey] then 
				local alt = currentDistance + (neighbor - current).Magnitude
				if alt < distances[neighborKey] then
					distances[neighborKey] = alt
					previous[neighborKey] = current
					openSet:push(neighbor, alt)
				end
			end
		end
	end
	
	if distances[goalKey] == math.huge then
		warn("Dijkstra pathfinding could not find a path")
		return nil
	end

	local path = {}
	local curr = grid[goalKey] -- // Start with goal position
	while curr do
		table.insert(path, 1, curr)
		local currKey = Vector3toString(curr)
		curr = previous[currKey]
	end

	return path
end
---------------------------------------------
-- // Visualization Functions
---------------------------------------------

-- // Enable or disable path visualization
function PathfindingModule:SetVisualization(enabled: boolean, options: table)
	self.visualization.enabled = enabled

	if options then
		for k, v in pairs(options) do
			if self.visualization[k] ~= nil then
				self.visualization[k] = v
			end
		end
	end

	if enabled and not self.visualFolder then
		self:CreateVisualizationFolder()
	end

	if not enabled and self.visualFolder then
		self:ClearVisualization()
	end

	return self
end

-- // Create folder to store visualization objects

function PathfindingModule:CreateVisualizationFolder()
	self.visualFolder = workspace:FindFirstChild("PathfindingVisualization") or Instance.new("Folder")
	
	if not self.visualFolder.Parent then
		self.visualFolder.Name = "PathfindingVisualization"
		self.visualFolder.Parent = workspace
	end
	
	return self.visualFolder
end

function PathfindingModule:VisualizePath(path, customOptions)
	if not path or #path == 0 then
		warn("Path is invalid or empty.")
		return
	end
	
	self:ClearVisualization()

	local options = {}
	for k, v in pairs(self.visualization) do
		options[k] = v
	end
	if customOptions then
		for k, v in pairs(customOptions) do
			options[k] = v
		end
	end

	-- // Create Folder
	if not self.visualFolder then
		self:CreateVisualizationFolder()
	end

	-- // Extract positions from waypoints
	local positions = {}
	for i, waypoint in ipairs(path) do
		local position

		if typeof(waypoint) == "Vector3" then
			position = waypoint
		elseif typeof(waypoint) == "Instance" then
			position = waypoint.Position
		elseif typeof(waypoint) == "PathWaypoint" then 
			position = waypoint.Position
		elseif type(waypoint) == "table" and waypoint.Position then
			position = waypoint.Position
		else
			warn("Waypoint at index " .. i .. " is not a recognized type. Type:", typeof(waypoint))
			continue
		end

		positions[i] = position
	end

	for i, position in ipairs(positions) do

		local point = Instance.new("Part")
		point.Name = "PathPoint_" .. i
		point.Shape = Enum.PartType.Ball
		point.Size = Vector3.new(options.pointSize, options.pointSize, options.pointSize)
		point.Position = position
		point.Anchored = true
		point.CanCollide = false
		point.Material = Enum.Material.Neon
		point.Transparency = 0.3
		point.Color = options.pointColor
		point.Parent = self.visualFolder

		if i < #positions then
			local nextPosition = positions[i + 1]
			local direction = (nextPosition - position)
			local distance = direction.Magnitude

			if distance > 0 then
				direction = direction.Unit
				local line = Instance.new("Part")
				line.Name = "PathLine_" .. i
				line.Size = Vector3.new(options.lineSize, options.lineSize, distance)

				local midpoint = position + direction * (distance / 2)
				line.CFrame = CFrame.lookAt(midpoint, nextPosition)

				line.Anchored = true
				line.CanCollide = false
				line.Material = Enum.Material.Neon
				line.Transparency = 0.5
				line.Color = options.lineColor
				line.Parent = self.visualFolder
			end
		end
	end

	return self
end

-- // Visualize grid points for debugging

function PathfindingModule:VisualizeGrid(grid, customOptions)
	if not grid then return end
	if not self.visualization.enabled then return end

	local options = {}
	for k, v in pairs(self.visualization) do
		options[k] = v
	end

	if customOptions then
		for k, v in pairs(customOptions) do
			options[k] = v
		end
	end

	if not self.visualFolder then
		self:CreateVisualizationFolder()
	end

	-- // Create a subfolder for the nodes
	
	local gridFolder = Instance.new("Folder")
	gridFolder.Name = "GridVisualization"
	gridFolder.Parent = workspace

	-- // Visualize the grid points
	
	local count = 0
	for pos, _ in pairs(grid) do
		count = count + 1

		local point = Instance.new("Part")
		point.Name = "GridPoint_" .. count
		point.Shape = Enum.PartType.Ball
		point.Size = Vector3.new(tonumber(options.pointSize) * 0.5, tonumber(options.pointSize) * 0.5, tonumber(options.pointSize) * 0.5)
		point.Position = stringToVector3(pos)
		point.Anchored = true
		point.CanCollide = false
		point.Material = Enum.Material.Neon
		point.Transparency = 0.7
		point.Color = Color3.fromRGB(100, 100, 100)
		point.Parent = gridFolder or workspace
	end

	return self
end

function PathfindingModule:ClearVisualization()
	local visualizationFolder = workspace:FindFirstChild("PathfindingVisualization")
	if visualizationFolder then
		visualizationFolder:ClearAllChildren()
	end

	if self.visualFolder then
		self.visualFolder:ClearAllChildren()
	end

	return self
end

function PathfindingModule:RemoveWaypointVisualizer(index) -- // Function to remove waypoints when reached
	if not self.visualFolder then return end
	
	local pointName = "PathPoint_" .. index
	local point = self.visualFolder:FindFirstChild(pointName)
	if point then
		point:Destroy()
	end
	
	local lineName = "PathLine_" .. index
	local line = self.visualFolder:FindFirstChild(lineName)
	if line then
		line:Destroy()
	end
end

---------------------------------------------
-- // Follow Path Function (Function to actually move the npc to the target)
---------------------------------------------

function PathfindingModule:FollowPath(character, path)
	if not path or #path == 0 then
		warn("Path is invalid or empty.")
		return
	end

	local humanoid = character:FindFirstChild("Humanoid")
	if not humanoid then
		warn("No humanoid found in character.")
		return
	end

	local rootPart = character:FindFirstChild("HumanoidRootPart")
	if not rootPart then
		warn("No HumanoidRootPart found in character.")
		return
	end

	-- // Clean up any existing connections, the reason we cant use self:StopFollowing() is since it also clears the path too
	
	if self.pathConnection then
		self.pathConnection:Disconnect()
		self.pathConnection = nil
	end

	if self.stateConnection then
		self.stateConnection:Disconnect()
		self.stateConnection = nil
	end

	self.currentWaypointIndex = 1
	
	local start = tick()
	
	local function moveToNextWaypoint()
		if self.currentWaypointIndex > 1 then
			self:RemoveWaypointVisualizer(self.currentWaypointIndex - 1)
		end

		if self.currentWaypointIndex > #path then
			humanoid:MoveTo(rootPart.Position)
			print("Path completed. Pathfollowing stopped.")
			print("Time taken: " .. tick() - start .. " seconds. ("..self.currentAlgorithm..")")
			self:ClearVisualization()
			self:StopFollowing()
			return
		end

		local currentWaypoint = path[self.currentWaypointIndex]

		-- // Jump Handler
		if typeof(currentWaypoint) == "PathWaypoint" and currentWaypoint.Action == Enum.PathWaypointAction.Jump then
			if humanoid.FloorMaterial ~= Enum.Material.Air then
				humanoid:ChangeState(Enum.HumanoidStateType.Jumping)
			end
		end

		-- // Move to current waypoint
		humanoid:MoveTo(typeof(currentWaypoint) == "Vector3" and currentWaypoint or currentWaypoint.Position)

		self.currentWaypointIndex += 1
	end

	self.pathConnection = humanoid.MoveToFinished:Connect(function(reached)
		if reached then
			moveToNextWaypoint()
		end
	end)

	-- // Start moving
	moveToNextWaypoint()
end

function PathfindingModule:StopFollowing()
	self.isFollowing = false
	if self.pathConnection then
		self.pathConnection:Disconnect()
		self.pathConnection = nil
	end
	if self.targetMonitor then
		self.targetMonitor:Disconnect()
		self.targetMonitor = nil
	end
	if self.stateConnection then
		self.stateConnection:Disconnect()
		self.stateConnection = nil
	end
	self:ClearVisualization()
end

return PathfindingModule
