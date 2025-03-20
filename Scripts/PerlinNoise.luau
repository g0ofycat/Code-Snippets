--[[ README:
	- // Perlin Terrain Generation, Includes Greedy Meshing. I'll probably update this to add more optomizations and features
--]]

---------------------------------------------
-- // Configuration
---------------------------------------------
local Settings = {
	Seed = math.random(1, 10000000), -- // Random seed
	Frequency = 4, -- // How frequent the hills are
	Power = 10, -- // How tall the hills are
	Resolution = 500, -- // How smooth or detailed the chunks are
	BlockSize = 2, -- // How big each block is (studs)
	ChunkSize = 64, -- // How big the chunks are
	RenderDistance = 4, -- // Number of chunks to render from center
	UseGreedyMeshing = true -- // Enable greedy meshing
}

---------------------------------------------
-- // Variables
---------------------------------------------
local Workspace = game:GetService("Workspace")
local RunService = game:GetService("RunService")
local Terrain = {}
local TerrainContainer = Instance.new("Folder")
TerrainContainer.Name = "GeneratedTerrain"
TerrainContainer.Parent = Workspace

local PartCache = {}

local function GetPart() -- // Reuses a part from the 'PartCache' or creates a new one with standard properties
	local part = table.remove(PartCache) or Instance.new("Part")
	part.Anchored = true
	part.Color = Color3.fromRGB(16, 178, 21)
	part.Size = Vector3.new(Settings.BlockSize, Settings.BlockSize, Settings.BlockSize)
	part.Parent = TerrainContainer
	return part
end

local noise = math.noise
local floor = math.floor
local abs = math.abs
local min = math.min

---------------------------------------------
-- // Terrain Generation
---------------------------------------------
local function GenerateHeightMap(chunkX, chunkZ) -- // Creates height values for terrain using multiple layers of noise
	local heightMap = {}
	local chunkOffsetX = chunkX * Settings.ChunkSize
	local chunkOffsetZ = chunkZ * Settings.ChunkSize

	for x = 0, Settings.ChunkSize - 1 do
		heightMap[x] = {}
		for z = 0, Settings.ChunkSize - 1 do
			local worldX = chunkOffsetX + x
			local worldZ = chunkOffsetZ + z

			local y1 = noise(
				(worldX * Settings.Frequency) / Settings.Resolution,
				(worldZ * Settings.Frequency) / Settings.Resolution,
				Settings.Seed
			)
			local y2 = noise(
				(worldX * Settings.Frequency * 0.125) / Settings.Resolution,
				(worldZ * Settings.Frequency * 0.125) / Settings.Resolution,
				Settings.Seed + 100
			)
			local y3 = noise(
				(worldX * Settings.Frequency * 4) / Settings.Resolution,
				(worldZ * Settings.Frequency * 4) / Settings.Resolution,
				Settings.Seed + 200
			)

			local height = floor((y1 * y2 * Settings.Power * Settings.Power) + y3 * Settings.Power)
			heightMap[x][z] = height
		end
	end

	return heightMap
end

---------------------------------------------
-- // Greedy Meshing Algorithm
---------------------------------------------
local function GreedyMesh(heightMap, chunkX, chunkZ) -- // Used if 'UseGreedyMeshing' is true, Combinies adjacent blocks of the same height into larger parts. (More Optomized)
	local chunkOffsetX = chunkX * Settings.ChunkSize
	local chunkOffsetZ = chunkZ * Settings.ChunkSize
	local blockSize = Settings.BlockSize

	local visited = {}
	for x = 0, Settings.ChunkSize - 1 do
		visited[x] = {}
		for z = 0, Settings.ChunkSize - 1 do
			visited[x][z] = false
		end
	end

	for startX = 0, Settings.ChunkSize - 1 do
		for startZ = 0, Settings.ChunkSize - 1 do
			if not visited[startX][startZ] then
				local height = heightMap[startX][startZ]

				local endX = startX
				while endX + 1 < Settings.ChunkSize and 
					not visited[endX + 1][startZ] and 
					heightMap[endX + 1][startZ] == height do
					endX = endX + 1
				end

				local endZ = startZ
				local canExpandZ = true

				while canExpandZ and endZ + 1 < Settings.ChunkSize do
					for x = startX, endX do
						if visited[x][endZ + 1] or heightMap[x][endZ + 1] ~= height then
							canExpandZ = false
							break
						end
					end

					if canExpandZ then
						endZ = endZ + 1
					end
				end

				for x = startX, endX do
					for z = startZ, endZ do
						visited[x][z] = true
					end
				end

				local sizeX = (endX - startX + 1) * blockSize
				local sizeZ = (endZ - startZ + 1) * blockSize

				local part = GetPart()
				part.Size = Vector3.new(sizeX, blockSize, sizeZ)

				local posX = (startX + (endX - startX) / 2) * blockSize + chunkOffsetX * blockSize
				local posY = height * blockSize
				local posZ = (startZ + (endZ - startZ) / 2) * blockSize + chunkOffsetZ * blockSize

				part.Position = Vector3.new(posX, posY, posZ)
			end
		end
	end
end

---------------------------------------------
-- // Block-by-Block Generation
---------------------------------------------
local function GenerateTraditional(heightMap, chunkX, chunkZ) -- // Used if 'UseGreedyMeshing' is false, generates each block seperatley. (Unoptomized since there would be tons of parts)
	local chunkOffsetX = chunkX * Settings.ChunkSize
	local chunkOffsetZ = chunkZ * Settings.ChunkSize
	local blockSize = Settings.BlockSize

	for x = 0, Settings.ChunkSize - 1 do
		for z = 0, Settings.ChunkSize - 1 do
			local height = heightMap[x][z]
			local part = GetPart()

			local posX = x * blockSize + chunkOffsetX * blockSize
			local posY = height * blockSize
			local posZ = z * blockSize + chunkOffsetZ * blockSize

			part.Position = Vector3.new(posX, posY, posZ)
		end
	end
end

---------------------------------------------
-- // Chunk Generation
---------------------------------------------
local function GenerateChunk(chunkX, chunkZ) -- // See what algorithm to use based on 'UseGreedyMeshing'
	local heightMap = GenerateHeightMap(chunkX, chunkZ)

	if Settings.UseGreedyMeshing then
		GreedyMesh(heightMap, chunkX, chunkZ)
	else
		GenerateTraditional(heightMap, chunkX, chunkZ)
	end

	Terrain[chunkX .. "," .. chunkZ] = true
end

---------------------------------------------
-- // Main Generation Function
---------------------------------------------
local function GenerateTerrain() -- // Main function that creates the spiral pattern of chunk generation around the center point
	TerrainContainer:ClearAllChildren()

	local centerChunkX = floor(Settings.Resolution / Settings.ChunkSize / 2)
	local centerChunkZ = floor(Settings.Resolution / Settings.ChunkSize / 2)

	local directions = {
		{0, 1}, -- // Right
		{1, 0}, -- // Down
		{0, -1}, -- // Left
		{-1, 0}  -- // Up
	}

	local chunkX, chunkZ = centerChunkX, centerChunkZ
	local direction = 0
	local stepsInDirection = 1
	local stepsTaken = 0
	local totalChunks = (Settings.RenderDistance * 2 + 1) ^ 2
	local chunksGenerated = 0

	GenerateChunk(chunkX, chunkZ)
	chunksGenerated = chunksGenerated + 1

	while chunksGenerated < totalChunks do
		chunkX = chunkX + directions[direction + 1][1]
		chunkZ = chunkZ + directions[direction + 1][2]

		if abs(chunkX - centerChunkX) <= Settings.RenderDistance and 
			abs(chunkZ - centerChunkZ) <= Settings.RenderDistance then
			GenerateChunk(chunkX, chunkZ)
			chunksGenerated = chunksGenerated + 1
		end

		stepsTaken = stepsTaken + 1

		if stepsTaken == stepsInDirection then
			stepsTaken = 0
			direction = (direction + 1) % 4

			if direction % 2 == 0 then
				stepsInDirection = stepsInDirection + 1
			end
		end
	end

	print("Terrain generation complete. Generated " .. chunksGenerated .. " chunks.")
end

GenerateTerrain()
