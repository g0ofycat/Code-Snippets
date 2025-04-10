--[[	README:
	
	- Turn StreamingEnabled off, the CreateWorldGrid relies on Raycasting which may not work if the grid and you are far away. This can lead to obstacles just not be rendered and raycasting passing through it
	- Includes Dijkstra, A*, Grid (A* but without diagonal vectors), and normal PathfindingService
	- Grid Based with Spatial Lookups
	- Visaulization Properties and more
	- This was made for a project and not really meant to be used for general use.
	- W.I.P
	
	Example Code:
	
-- // Do calculations on the server if you want to move npcs
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")
local UserInputService = game:GetService("UserInputService")
local PathfindingModule = require(ReplicatedStorage:WaitForChild("PathfindingModule"))

local pathfinder = PathfindingModule.new()

pathfinder:SetVisualization({ -- // Set to true to visualize grid, if youre making massive grids then don't set this to true
	pointSize = 1, -- // Size of the Visual Part of the nodes
	lineSize = 0.2, -- // Line Size of the Path Part
	visualizeGrid = true, -- // Visualize the Grid
	visualizeLine = true, -- // Visualize the Line
	pointColor = Color3.fromRGB(102, 0, 255), -- // Color of the Visual Part
	lineColor = Color3.fromRGB(136, 0, 255), -- // Color of the Grid Part
	pointColorGrid = Color3.fromRGB(255, 255, 255) -- // Color of the Visual Part of Grid
})

local player = Players.LocalPlayer
local character = player.Character or player.CharacterAdded:Wait()

local grid = pathfinder:CreateWorldGrid(Vector3.new(0, 0, 0), Vector3.new(200, 100, 200), 2, 2) -- // Center: Center of the grid, Size: Size of the grid. Resoulution: Distance between each point (Recommended: 2 since it needs to detect collision). BufferDistance: How far away each point is from the wall, this is to stop pathfinding sticking to the walls
-- // Roblox Pathfinding's AgentRadius is based on bufferDistance. If you want it to be different then change the PathfindingModule:RobloxPathfinding() function

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
				{"Grid", "Roblox"} -- // Possible Names: Roblox (PathfindingService), AStar (A* Pathfinding Algorithm), Dijkstra (Dijkstra Pathfinding Algorithm), Grid (AStar without the diagonal vecotrs, moves in a grid-like pattern)
				-- // The table above can store multiple pathfinding system as a fallback, if A* Fails then the normal Roblox one can be used. There can be as much as you want.
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
-- // Constructor
---------------------------------------------
function PathfindingModule.new()
	local self = setmetatable({}, PathfindingModule)
	-- // Holder Variables, will change
	self.currentAlgorithm = nil
	self.resolution = 2
	self.hasJumped = false
	self.bufferDistance = 2
	self.visualizeLine = false
	self.snapDistance = 5 -- // You might want to multiply it by the resolution for spaced grids

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
			return self:AStarPathfinding(start, goal, grid)  -- // Just A* without the diagonal vectors
		end
	}

	-- // Visualization properties

	self.visualFolder = nil

	self.visualization = { -- // Holder variables, wont actually be used for visualization
		enabled = false,
		pointSize = 0.5,
		lineSize = 0.2,
		pointColor = Color3.fromRGB(212, 0, 255),
		lineColor = Color3.fromRGB(212, 0, 255),
		pointColorGrid = Color3.fromRGB(212, 0, 255)
	}

	return self
end

-- // Find a path using the best available algorithm

function PathfindingModule:FindBestPath(start: Vector3, goal: Vector3, grid: table, algorithmNames: table)
	for _, name in ipairs(algorithmNames) do
		local path = self.algorithms[name] and self.algorithms[name](start, goal, grid)
		if path then 
			if self.visualizeLine then -- // We dont check for the self visualization as those are 2 seperate visualizations
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
local function Vector3toString(vector: Vector3): string
	return tostring(vector.X) .. "," .. tostring(vector.Y) .. "," .. tostring(vector.Z)
end

local function heuristic(a: number, b: number): number -- // Euclidean-distance Heuristic, maybe use Manhattan for Grid Pathfinding
	return (b - a).Magnitude
end

local function stringToVector3(str: string): Vector3?
	local x, y, z = str:match("([^,]+),([^,]+),([^,]+)")
	if x and y and z then
		return Vector3.new(tonumber(x), tonumber(y), tonumber(z))
	end
	return nil
end

local raycastParams = RaycastParams.new()
raycastParams.IgnoreWater = true
raycastParams.FilterType = Enum.RaycastFilterType.Exclude

---------------------------------------------
-- // GetNeighbors Function, used to get the neighbors of a node which lays out obstacles and sees if you can go to certain nodes. Basically shows where the pathfinder needs to go.
-- // Time Complexity: O(1). Edge Cases: O(k). 'k' being the number of neighbors. For large values of k this can be expensive, so we opt to only call this function when we need to.
---------------------------------------------
function PathfindingModule:GetNeighbors(position: Vector3, grid: table, resolution: number): table
	resolution = resolution or self.resolution
	local neighbors = {}

	if not self.neighborOffsets or self.neighborOffsets.resolution ~= resolution or
		(self.neighborOffsets.algorithm ~= self.currentAlgorithm) then

		local offsets = {}
		if self.currentAlgorithm == "Grid" then
			-- // No diagonals
			table.insert(offsets, Vector3.new(resolution, 0, 0))  -- // Right
			table.insert(offsets, Vector3.new(-resolution, 0, 0)) -- // Left
			table.insert(offsets, Vector3.new(0, 0, resolution))  -- // Forward
			table.insert(offsets, Vector3.new(0, 0, -resolution)) -- // Backward
		else
			-- // Include diagonals for other algorithms
			table.insert(offsets, Vector3.new(resolution, 0, 0))
			table.insert(offsets, Vector3.new(-resolution, 0, 0))
			table.insert(offsets, Vector3.new(0, 0, resolution))
			table.insert(offsets, Vector3.new(0, 0, -resolution))
			table.insert(offsets, Vector3.new(resolution, 0, resolution))
			table.insert(offsets, Vector3.new(resolution, 0, -resolution))
			table.insert(offsets, Vector3.new(-resolution, 0, resolution))
			table.insert(offsets, Vector3.new(-resolution, 0, -resolution))
		end

		self.neighborOffsets = {
			offsets = offsets,
			resolution = resolution,
			algorithm = self.currentAlgorithm
		}
	end

	local cachedOffsets = self.neighborOffsets.offsets
	local posKey = Vector3toString(position)

	if grid.spatialLookup then
		local searchRadius = resolution * 1.5 -- // Slightly larger to catch neighbors
		local nearbyPoints = grid.spatialLookup:GetPointsInRadius(position, searchRadius)

		for neighborKey, neighborPos in pairs(nearbyPoints) do
			if neighborKey ~= posKey then
				local direction = neighborPos - position
				local distance = direction.Magnitude

				if distance <= resolution * 1.5 then
					if self.currentAlgorithm == "Grid" then
						local isCardinal = (direction.X == 0 or direction.Z == 0) and
							(math.abs(direction.X) == resolution or math.abs(direction.Z) == resolution)
						if not isCardinal then continue end
					end

					local rayOrigin = position + Vector3.new(0, 1, 0)
					local rayResult = workspace:Raycast(rayOrigin, direction, raycastParams)

					if not rayResult then
						table.insert(neighbors, neighborPos)
					end
				end
			end
		end
	else
		for _, offset in ipairs(cachedOffsets) do
			local neighborPos = position + offset
			local neighborKey = Vector3toString(neighborPos)

			if grid[neighborKey] then
				local direction = offset
				local rayOrigin = position + Vector3.new(0, 1, 0)
				local rayResult = workspace:Raycast(rayOrigin, direction, raycastParams)

				if not rayResult then
					table.insert(neighbors, grid[neighborKey])
				end
			end
		end
	end

	return neighbors
end

local function newBinaryHeap(): table
	local heap = {size = 0}
	local data = {}

	local function swap(i, j) -- // Swap the 2 parameters
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
-- // Time Complexity for PathfindingService is unknown
---------------------------------------------
function PathfindingModule:RobloxPathfinding(start: Vector3, goal: Vector3) -- // I don't really think this needs documentation
	local path = PathfindingService:CreatePath({
		AgentRadius = self.bufferDistance,
		AgentHeight = 5,
		WaypointSpacing = self.resolution,
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
-- // Time Complexity: O(b^d). b is the Branching Factor and d is the depth. With Binary Heap: O((E + V) log V)
---------------------------------------------
function PathfindingModule:AStarPathfinding(start: Vector3, goal: Vector3, grid: table)
	if not grid or not next(grid) then
		if self.currentAlgorithm == "Grid" then
			warn("Grid pathfinding requires a valid grid")
		else
			warn("AStar pathfinding requires a valid grid")
		end
		return nil
	end

	if start == goal then
		return {start} -- // If we're already at the goal then we can just return the start
	end

	local startKey = Vector3toString(start)
	local goalKey = Vector3toString(goal)

	-- // Try to snap start and goal to nearest grid positions if they're not exactly on grid points
	if not grid[startKey] then -- // Snaps to closest point (Start)
		local closestPos
		local closestDist = math.huge

		if grid.spatialLookup then
			closestPos, closestDist = grid.spatialLookup:FindNearestPoint(start)
		else
			for key, pos in pairs(grid) do
				local dist = (pos - start).Magnitude
				if dist < closestDist then
					closestDist = dist
					closestPos = pos
				end
			end
		end

		if closestPos and closestDist < self.snapDistance then
			start = closestPos
			startKey = Vector3toString(start)
		else
			warn("Start position not in grid and no nearby point found: ", startKey)
			return nil
		end
	end

	if not grid[goalKey] then
		local closestPos
		local closestDist = math.huge

		if grid.spatialLookup then
			closestPos, closestDist = grid.spatialLookup:FindNearestPoint(goal)
		else
			for key, pos in pairs(grid) do
				local dist = (pos - goal).Magnitude
				if dist < closestDist then
					closestDist = dist
					closestPos = pos
				end
			end
		end

		if closestPos and closestDist < self.snapDistance then
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

	while not openSet:isEmpty() do
		local current = openSet:pop() -- // Node with lowest fScore
		local currentKey = Vector3toString(current)

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

---------------------------------------------
-- // World Grid Creation, it works by generating points based on directional vectors and then storing them into a table. It creates points in a box shaped region using the X and Z coordinates (spaced with the resolution parameter) in which is then organized into a 2D spacial grid.
-- // Uses Raycasting to detect heights of obstacles and spaces nodes from it using the bufferDistance arugment. Center is the position of the region and the size is the size of the region.
-- // Time Complexity (Spatial Lookups): O(1). Time Complexity is crucial since if this is slow then all of the other pathfinding algorithms are slower.
-- // Time Complexity (Grid Generation): O(n).
---------------------------------------------

function PathfindingModule:CreateWorldGrid(center: Vector3, size: Vector3, resolution: number, bufferDistance: number): table
	if size.Y == 0 then
		warn("size.Y is small (" .. size.Y .. "); consider increasing for uneven terrain")
	end
	self.bufferDistance = bufferDistance
	self.resolution = resolution
	local snapDist = self.snapDistance

	local grid = {}

	local spatialGrid = {}
	local cellSize = resolution * 2
	local function getCellKey(x, z)
		return math.floor(x / cellSize) .. "," .. math.floor(z / cellSize)
	end

	local halfSize = size * 0.5
	local minPos = center - halfSize
	local maxPos = center + halfSize

	local rayStartHeight = maxPos.Y + 100
	local downVector = Vector3.new(0, -size.Y - 100, 0)
	local upOffset = Vector3.new(0, 1, 0)
	local gridPoints = 0

	local directions = {
		Vector3.new(bufferDistance, 0, 0),
		Vector3.new(-bufferDistance, 0, 0),
		Vector3.new(0, 0, bufferDistance),
		Vector3.new(0, 0, -bufferDistance)
	}

	local raycastParams = RaycastParams.new()
	raycastParams.FilterType = Enum.RaycastFilterType.Exclude -- // Might want to exclude characters, although errors on server script that require this module

	-- // Batch process grid points 
	for x = minPos.X, maxPos.X, resolution do
		local roundedX = math.round(x / resolution) * resolution
		for z = minPos.Z, maxPos.Z, resolution do
			local roundedZ = math.round(z / resolution) * resolution
			local pos
			local top = Vector3.new(roundedX, rayStartHeight, roundedZ)
			local result = workspace:Raycast(top, downVector, raycastParams)
			if not result then continue end
			pos = Vector3.new(roundedX, result.Position.Y, roundedZ)

			local rayOrigin = pos + upOffset
			local isTooCloseToWall = false

			for _, direction in directions do
				local rayResult = workspace:Raycast(rayOrigin, direction, raycastParams)
				if rayResult and (rayResult.Position - rayOrigin).Magnitude < bufferDistance then
					isTooCloseToWall = true
					break
				end
			end

			if not isTooCloseToWall then
				local posKey = Vector3toString(pos)
				grid[posKey] = pos

				local cellKey = getCellKey(roundedX, roundedZ)
				if not spatialGrid[cellKey] then
					spatialGrid[cellKey] = {}
				end
				spatialGrid[cellKey][posKey] = pos
				gridPoints = gridPoints + 1
			end
		end
	end

	-- // Attach spatial lookup methods
	grid.spatialLookup = {
		_spatialGrid = spatialGrid,
		_cellSize = cellSize,

		FindNearestPoint = function(self, position)
			local baseCellX = math.floor(position.X / self._cellSize)
			local baseCellZ = math.floor(position.Z / self._cellSize)
			local closestPoint = nil
			local closestDist = math.huge

			for searchRadius = 0, snapDist do
				local pointsFound = false

				for xOffset = -searchRadius, searchRadius do
					for zOffset = -searchRadius, searchRadius do
						if searchRadius == 0 or math.abs(xOffset) == searchRadius or math.abs(zOffset) == searchRadius then
							local cellKey = (baseCellX + xOffset) .. "," .. (baseCellZ + zOffset)
							local cell = self._spatialGrid[cellKey]

							if cell then
								pointsFound = true
								for _, pos in pairs(cell) do
									local dist = (position - pos).Magnitude
									if dist < closestDist then
										closestDist = dist
										closestPoint = pos
									end
								end
							end
						end
					end
				end

				if pointsFound and closestPoint then
					return closestPoint, closestDist
				end
			end

			return nil
		end,

		GetPointsInRadius = function(self, position, radius)
			local pointsInRadius = {}
			local cellRadius = math.ceil(radius / self._cellSize)
			local baseCellX = math.floor(position.X / self._cellSize)
			local baseCellZ = math.floor(position.Z / self._cellSize)

			for xOffset = -cellRadius, cellRadius do
				for zOffset = -cellRadius, cellRadius do
					local cellKey = (baseCellX + xOffset) .. "," .. (baseCellZ + zOffset)
					local cell = self._spatialGrid[cellKey]

					if cell then
						for key, pos in pairs(cell) do
							if (position - pos).Magnitude <= radius then
								pointsInRadius[key] = pos
							end
						end
					end
				end
			end

			return pointsInRadius
		end,

		IsValidPosition = function(self, position, threshold)
			threshold = threshold or self._cellSize
			local nearestPoint, distance = self:FindNearestPoint(position)
			return nearestPoint ~= nil and distance <= threshold
		end
	}

	print("Created world grid with " .. gridPoints .. " points")

	if self.visualization and self.visualization.enabled then
		self:VisualizeGrid(grid)
	end

	return grid
end


---------------------------------------------
-- // Dijkstra Pathfinder (Used for mainly finding the shortest path between 2 nodes, more costly)
-- // Time Complexity: O((E + V) log V). V is the number of nodes, E is connections	/ edges
---------------------------------------------
function PathfindingModule:DijkstraPathfinding(start: Vector3, goal: Vector3, grid: table)
	if not grid or not next(grid) then
		warn("Dijkstra pathfinding requires a valid grid")
		return nil
	end

	local startKey = Vector3toString(start)
	local goalKey = Vector3toString(goal)

	-- // Snap start to nearest grid point if not found
	if not grid[startKey] then
		local closestPos
		local closestDist = math.huge

		if grid.spatialLookup then
			closestPos, closestDist = grid.spatialLookup:FindNearestPoint(start)
		else
			for key, pos in pairs(grid) do
				local dist = (pos - start).Magnitude
				if dist < closestDist then
					closestDist = dist
					closestPos = pos
				end
			end
		end

		if closestPos and closestDist < self.snapDistance then
			start = closestPos
			startKey = Vector3toString(start)
		else
			warn("Start position not found in grid and no nearby point found: ", startKey)
			return nil
		end
	end

	-- // Snap goal to nearest grid point if not found
	if not grid[goalKey] then
		local closestPos
		local closestDist = math.huge

		if grid.spatialLookup then
			closestPos, closestDist = grid.spatialLookup:FindNearestPoint(goal)
		else
			for key, pos in pairs(grid) do
				local dist = (pos - goal).Magnitude
				if dist < closestDist then
					closestDist = dist
					closestPos = pos
				end
			end
		end

		if closestPos and closestDist < self.snapDistance then
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

-- // Enable or disable path visualization and set the parameters

function PathfindingModule:SetVisualization(options: table)
	self.visualization.enabled = options.visualizeGrid

	if options then
		for k, v in pairs(options) do
			if self.visualization[k] ~= nil then
				self.visualization[k] = v
			end
		end
	end

	if options.visualizeLine == true and not self.visualFolder then
		self.visualizeLine = true
		self:CreateVisualizationFolder()
	end

	if not options.visualizeGrid and self.visualFolder then
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

function PathfindingModule:VisualizePath(path: table, customOptions: table?)
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

-- // Visualize grid points for debugging. Time Complexity: O(n)
function PathfindingModule:VisualizeGrid(grid: table, customOptions: table?) -- // We will use batching to create a certain amount of parts per frame so we don't lag

	if not self.visualization.enabled then
		warn("Visualization is disabled - call :SetVisualization({visualizeGrid = true}) first")
		return
	end

	if not grid or not next(grid) then
		warn("Grid is nil or empty - check your grid creation")
		return
	end

	local options = {
		pointSize = (self.visualization.pointSize or 0.5) * 0.5,
		pointColorGrid = self.visualization.pointColorGrid,
		transparency = 0.7
	}

	if customOptions then
		for k, v in pairs(customOptions) do
			options[k] = v
		end
	end

	-- // Clean up previous grid visualization if it exists
	if self.gridVisualFolder then
		if self.gridVisualizationRoutine then
			task.cancel(self.gridVisualizationRoutine)
			self.gridVisualizationRoutine = nil
		end
		self.gridVisualFolder:Destroy()
	end

	self.gridVisualFolder = Instance.new("Folder")
	self.gridVisualFolder.Name = "GridVisualization"
	self.gridVisualFolder.Parent = workspace

	local gridPointsCount = 0
	for key, value in pairs(grid) do
		if typeof(value) == "Vector3" then
			gridPointsCount = gridPointsCount + 1
		end
	end
	print("Attempting to visualize", gridPointsCount, "grid points")

	if gridPointsCount == 0 then
		warn("No valid grid points found to visualize.")
		return 
	end

	self.gridVisualizationRoutine = task.spawn(function()
		local pointsCreated = 0
		local startTime = os.clock()

		for posKey, pos in pairs(grid) do
			if typeof(pos) == "Vector3" then
				pointsCreated += 1
				local position = pos

				local point = Instance.new("Part")
				point.Name = "GridPoint_" .. pointsCreated
				point.Shape = Enum.PartType.Ball
				point.Size = Vector3.new(options.pointSize, options.pointSize, options.pointSize)
				point.Position = position
				point.Anchored = true
				point.CanCollide = false
				point.CanQuery = false
				point.CanTouch = false
				point.Material = Enum.Material.Neon
				point.Transparency = options.transparency
				point.Color = options.pointColorGrid
				point.Parent = self.gridVisualFolder

				if pointsCreated % 500 == 0 then
					task.wait()
				end
			end
		end

		self.gridVisualizationRoutine = nil
		print(string.format("Successfully visualized %d points in %.2f seconds",
			pointsCreated, os.clock() - startTime))
	end)
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

function PathfindingModule:RemoveWaypointVisualizer(index: number) -- // Function to remove waypoints when each node is reached
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
-- // Time Complexity: O(n)
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

	if self.heartbeatConnection then
		self.heartbeatConnection:Disconnect()
		self.heartbeatConnection = nil
	end

	self.currentWaypointIndex = 1

	local start = tick()

	local startPerNode = tick()

	self.isFollowing = true

	local function moveToNextWaypoint()
		if not self.isFollowing then return end

		if self.currentWaypointIndex > 1 then
			self:RemoveWaypointVisualizer(self.currentWaypointIndex - 1)
		end

		if self.currentWaypointIndex > #path then -- // End pathfinding
			humanoid:MoveTo(rootPart.Position)
			print("Path completed. Pathfollowing stopped.")
			print("Time taken: " .. tick() - start .. " seconds. ("..self.currentAlgorithm..")")
			self:StopFollowing()
			return
		end

		local currentWaypoint = path[self.currentWaypointIndex]

		-- // Jump Handler
		local function HumanoidJump()
			if typeof(currentWaypoint) == "PathWaypoint" and currentWaypoint.Action == Enum.PathWaypointAction.Jump then
				if humanoid.FloorMaterial ~= Enum.Material.Air then
					humanoid:ChangeState(Enum.HumanoidStateType.Jumping)
				end
			end
		end

		HumanoidJump()

		-- // Move to current waypoint
		humanoid:MoveTo(typeof(currentWaypoint) == "Vector3" and currentWaypoint or currentWaypoint.Position)

		self.currentWaypointIndex += 1
	end

	-- // Prevent the charcter from being stuck by jumping, you can also just stop following too. Made for a very specific edge case scenario where you close to a wall and try to go on top of it
	-- // It does this by tracking how long it took to go to each node using tick()

	self.heartbeatConnection = RunService.Heartbeat:Connect(function()
		local timeElapsed = tick() - startPerNode
		local maxTimeBetweenNodes = 0.5 * self.resolution
		if timeElapsed >= maxTimeBetweenNodes then
			if humanoid.FloorMaterial ~= Enum.Material.Air and self.hasJumped == false then
				print('Path obstructed. Now jumping.')
				startPerNode = tick()
				humanoid:ChangeState(Enum.HumanoidStateType.Jumping) -- // We can't use the  HumanoidJump() function as it is out of the scope
				self.hasJumped = true -- // hasJumped only function is to jump once, if it already has jumped after the next timeElapsed then the path is disconnected
			elseif self.hasJumped == true then
				warn("Cannot find a path even after jumping, Disconnecting.")
				self:StopFollowing()
			end
		end
	end)

	self.pathConnection = humanoid.MoveToFinished:Connect(function(reached)
		if reached then
			startPerNode = tick()
			moveToNextWaypoint() -- // Go to next node
		end
	end)

	-- // Start moving
	moveToNextWaypoint()
end

function PathfindingModule:StopFollowing()
	self.isFollowing = false
	self.hasJumped = false
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
	if self.heartbeatConnection then
		self.heartbeatConnection:Disconnect()
		self.heartbeatConnection = nil
	end
	self:ClearVisualization()
end

return PathfindingModule
