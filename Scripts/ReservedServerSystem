--[[

	Overview:
	
- This script allows for Reserved Server Creation

- Works by creating a string value with all necessary information, sending it out to all servers

- Uses Messenger to broadcast all information without a limit and is overall useful for higher CCU games (~10k+). Forum on Messenger: https://devforum.roblox.com/t/messenger-wrapper-for-messagingservice/1669313

- MemoryStoreService used to broadcast and used as a backup for all information (Especially each seperate player count for each reserved server)

- Main limit for higher CCU games is MemoryStoreService due to the amount of requests, You could use multiple hashmaps

- Estimated CCU Limit for Servers: ~50,000 (1,000 x game.Players.MaxPlayers)

--]]

-- // Variables

local serversFolder = game.ReplicatedStorage:WaitForChild("Servers")
local Messenger = require(game.ReplicatedStorage.Modules:WaitForChild("Messenger"))
local events = game.ReplicatedStorage:WaitForChild("Events")
local CreateServer = events:WaitForChild("CreateServer")
local TeleportService = game:GetService("TeleportService")
local PlaceId = game.PlaceId
local Players = game:GetService('Players')
local ServerInfo = Messenger.new("ServerInfo")
local MessagingService = game:GetService("MessagingService")
local JobID = game.JobId
local MemoryStoreService = game:GetService('MemoryStoreService')
local Hashmap = MemoryStoreService:GetSortedMap('PlayersInServers')
local Teleport = events:WaitForChild("Teleport")
local TextService = game:GetService("TextService")
local GetCurrentReservedIDEvent = game.ReplicatedStorage.Events.ReturnReservedID
local RunService = game:GetService("RunService")
local TextChatService = game:GetService("TextChatService")
local DBMessages = game.ReplicatedStorage.Events.DebugMessages

local serverEntries = {} -- // All Servers inside of a table

local serverDataCache = {} -- // Table to Optomize the amount of MemoryStoreService calls, Used to quickly acess data instead of just constantly calling MemoryStoreService for the data

local Config = {
	PLAYER_UPDATE_INTERVAL = math.max(5, math.floor(#Players:GetPlayers() / 10)), -- // (In Seconds) How fast the servers will update (Only for player count since its the only one that really changes, may need to dynamically scale it based on the PlayerCount, another option is to only call it when the PlayerCount Changes)
	Expiration = 86400, -- // The expiration of the hashmap in seconds (1 Day)
	SERVER_VALUE_FORMAT = "%s %d %s %s %s %d", -- // Format: code, playerCount, name, desc, vcMode, creatorId
	MAX_SERVERS_TO_LOAD = math.min(200, #serverEntries + 10), -- // Maximum amount of servers you can use in GetRangeAsync (Maximum: 200)
	LocalExpiration = 30 -- // (In Seconds) Optional data used for the serverDataCache table. The script will use the data in the table for (int) amount of time before resorting to calling MemoryStoreService again
}

local lastUpdate = time()
local pendingUpdate = false

-- // Teleport Function

local function teleportPlayer(player, placeId, reservedId) -- // Function to teleport users that press the join button, this includes the creator.
	if not player or not placeId or not reservedId then
		warn("Invalid arguments provided to teleportPlayer.")
		return false
	end
	local success, err = pcall(function()
		TeleportService:TeleportToPrivateServer(placeId, reservedId, {player}, nil, {reservedId})
	end)

	if not success then 
		Hashmap:RemoveAsync(reservedId) 
		warn("Teleport failed! Server: "..reservedId.." Error: "..tostring(err))
		return false
	end

	return success
end

-- // Creates or updates a StringValue instance for a server.

local function createServerEntry(code, playerCount, name, desc, vcMode, creatorId)
	if name == "[N/A]" and (not creatorId or creatorId == 0) or not code then return nil end -- // Prevent faulty servers with nil names and no creator

	local existingEntry = serverEntries[code]

	if existingEntry and existingEntry.Parent then
		existingEntry.Value = string.format(Config.SERVER_VALUE_FORMAT, code, playerCount, name, desc, tostring(vcMode), creatorId or 0)
		return existingEntry
	else

		local serverValue = Instance.new("StringValue")
		serverValue.Name = "Server: " .. code
		serverValue.Value = string.format(Config.SERVER_VALUE_FORMAT, code, playerCount, name, desc, tostring(vcMode), creatorId or 0)
		serverValue.Parent = serversFolder
		serverEntries[code] = serverValue

		return serverValue
	end
end

-- // Server creation event, handles the text filtering, creating string values, hashmap creation

CreateServer.OnServerEvent:Connect(function(player, Desc, Name, VCMode)
	local success, code = pcall(TeleportService.ReserveServer, TeleportService, PlaceId)

	if not success then
		warn("Failed to Reserve Server! "..tostring(code))
		return
	end

	local filteredName, filteredDesc

	local successName, resultName = pcall(function()
		return TextService:FilterStringAsync(Name, player.UserId)
	end)
	if successName and resultName then
		filteredName = resultName:GetNonChatStringForBroadcastAsync()
	else
		filteredName = "[Filtered]"
		warn("Failed to filter Name:", Name)
	end

	local successDesc, resultDesc = pcall(function()
		return TextService:FilterStringAsync(Desc, player.UserId)
	end)
	if successDesc and resultDesc then
		filteredDesc = resultDesc:GetNonChatStringForBroadcastAsync()
	else
		filteredDesc = "[Filtered]"
		warn("Failed to filter Desc:", Desc)
	end

	-- // Attempt to update player count safely
	
	local function updatePlayerCount()
		local updateSuccess, err = pcall(function()
			Hashmap:UpdateAsync(code, function(serverData)
				if not serverData then 
					-- // Key doesn't exist yet, initialize with 1 player which is the creator
					return {
						PlayerCount = 1, 
						Name = filteredName,
						Desc = filteredDesc,
						VCMode = VCMode,
						CreatorId = player.UserId
					} 
				end

				serverData.PlayerCount = (serverData.PlayerCount or 0) + 1
				return serverData
			end, Config.Expiration)
		end)

		if not updateSuccess then
			warn("UpdateAsync failed for", code, "Error:", err)

			local setSuccess, setErr = pcall(function()
				Hashmap:SetAsync(code, {
					PlayerCount = 1, 
					Name = filteredName,
					Desc = filteredDesc,
					VCMode = VCMode,
					CreatorId = player.UserId
				}, Config.Expiration)
			end)

			if not setSuccess then
				warn("SetAsync also failed for", code, "Error:", setErr)
			end
		end
	end

	updatePlayerCount()

	-- // Create a server entry before teleporting
	createServerEntry(code, 1, filteredName, filteredDesc, tostring(VCMode), player.UserId)

	-- // Broadcast the new server info
	local success, errPublish = pcall(function()
		ServerInfo:PublishAsync({
			Code = code,
			Name = filteredName,
			Desc = filteredDesc,
			VCMode = VCMode,
			CreatorId = player.UserId,
			PlayerCount = 1,
			Type = "NewServer"
		})
	end)

	if not success then
		warn("Failed to publish server data! "..tostring(errPublish))
	end

	-- // Teleport the player to the new server
	teleportPlayer(player, PlaceId, code)
end)


-- // Server Info Handling, every ServerInfo:PublishAsync has a Type value that is recieved by the PublishAsync handler. It could update the player count, create a new server on the hashmap (CreateServerEntry already handles StringValue creation), and remove the StringValue and the Hashmap

local function getCachedServerData(code)
	if Config.LocalExpiration == 0 then return nil end

	local cachedData = serverDataCache[code]
	if cachedData and (tick() - cachedData.timestamp) < Config.LocalExpiration then
		return cachedData.data
	end

	-- // If we have a cached entry but it's expired, we to update it
	
	local success, serverData = pcall(Hashmap.GetAsync, Hashmap, code)
	if success and serverData then
		serverDataCache[code] = {
			data = serverData,
			timestamp = tick()
		}
		return serverData
	elseif success and not serverData and cachedData then
		serverDataCache[code] = nil
		return nil
	elseif not success and cachedData then
		-- // On MemoryStore error, we could still use stale data with a warning
		warn("MemoryStore error, using stale data for "..code)
		return cachedData.data
	end

	return nil
end

function PlayerCountFunction(dataParam)
	if serverEntries[dataParam.Code] and serverEntries[dataParam.Code].Parent then
		local serverData = getCachedServerData(dataParam.Code)

		if serverData then
			local playerCount = dataParam.PlayerCount or serverData.PlayerCount or 0
			local name = serverData.Name or "[N/A]"
			local desc = serverData.Desc or "[No Description]"
			local vcMode = tostring(serverData.VCMode or false)
			local creatorId = serverData.CreatorId or 0

			-- // Update the StringValue
			serverEntries[dataParam.Code].Value = string.format(Config.SERVER_VALUE_FORMAT, 
				dataParam.Code, playerCount, name, desc, vcMode, creatorId)

			-- // Update the cache with the new player count
			if serverDataCache[dataParam.Code] then
				serverDataCache[dataParam.Code].data.PlayerCount = playerCount
				serverDataCache[dataParam.Code].timestamp = tick()
			end
		end
	end
end

function NewServerFunction(dataParam)
	if serverEntries[dataParam.Code] and serverEntries[dataParam.Code].Parent then
		local playerCount = dataParam.PlayerCount or 1
		local name = dataParam.Name or "[N/A]"
		local desc = dataParam.Desc or "[No Description]"
		local vcMode = tostring(dataParam.VCMode or false)
		local creatorId = dataParam.CreatorId or 0

		-- // Update the StringValue
		serverEntries[dataParam.Code].Value = string.format(Config.SERVER_VALUE_FORMAT, 
			dataParam.Code, playerCount, name, desc, vcMode, creatorId)

		-- // Update or create cache entry
		serverDataCache[dataParam.Code] = {
			data = {
				PlayerCount = playerCount,
				Name = name,
				Desc = desc,
				VCMode = vcMode == "true",
				CreatorId = creatorId
			},
			timestamp = tick()
		}
	end
end

function RemoveServerFunction(dataParam)
	if serverEntries[dataParam.Code] then
		serverEntries[dataParam.Code]:Destroy()
		serverEntries[dataParam.Code] = nil
		serverDataCache[dataParam.Code] = nil
		
		local success, err = pcall(function()
			Hashmap:RemoveAsync(dataParam.Code)
		end)
	end
end

-- // Handles the type of the Server, Basically a manager to create, update, and remove servers when sent using :PublishAsync()

ServerInfo:SubscribeAsync(function(data)
	if not data or not data.Code then return end

	local cachedData = serverDataCache[data.Code]
	local useCache = cachedData and (tick() - cachedData.timestamp) < Config.LocalExpiration

	if data.Type == "PlayerCountUpdate" then
		if useCache then
			cachedData.data.PlayerCount = data.PlayerCount
			cachedData.timestamp = tick()
			PlayerCountFunction(data)
		else
			PlayerCountFunction(data)
		end
	elseif data.Type == "NewServer" or not data.Type then
		NewServerFunction(data)
	elseif data.Type == "RemoveServer" then
		RemoveServerFunction(data)
	end
end)

-- // Creates StringValue instances for each server in the provided Hashmap data. Checks values before procedding to use the createServerEntry which Creates the StringValues
-- // Each entry should contain:
-- // key (string): The server code.
-- // value (table): The server data (PlayerCount, Name, Desc, VCMode, CreatorId).

local function CreateLoadedValues(HashmapForServers)
	for _, entry in ipairs(HashmapForServers) do
		local serverData = entry.value
		if serverData and type(serverData) == "table" then
			local playerCount = serverData.PlayerCount or 1

			if playerCount > 0 then
				createServerEntry(
					entry.key, 
					playerCount,	
					serverData.Name or "[N/A]", 
					serverData.Desc or "[No Description]", 
					tostring(serverData.VCMode or false), 
					serverData.CreatorId or 0
				)
			end
		end
	end
end

-- // Fetches all reserved servers from MemoryStoreService and updates the serverEntries table.
-- // Removes invalid server entries that no longer exist in MemoryStoreService.
-- // Calls CreateLoadedValues to create new server entries for valid servers.

local function LoadReservedServers()
	local success, allServers = pcall(function()
		return Hashmap:GetRangeAsync(Enum.SortDirection.Ascending, Config.MAX_SERVERS_TO_LOAD)
	end)

	if not success then 
		warn("Failed to load reserved servers! " .. tostring(allServers))
		return 
	end

	local validServerCodes = {}
	for _, entry in ipairs(allServers) do
		validServerCodes[entry.key] = true

		serverDataCache[entry.key] = {
			data = entry.value,
			timestamp = tick()
		}
	end

	for code, serverValue in pairs(serverEntries) do
		if not validServerCodes[code] then
			serverValue:Destroy()
			serverEntries[code] = nil
			serverDataCache[code] = nil
		end
	end

	CreateLoadedValues(allServers)
end

LoadReservedServers()

task.spawn(function()
	while true do
		task.wait(Config.PLAYER_UPDATE_INTERVAL)
		LoadReservedServers() -- // Update Servers (Mainly player count since it's the only changing value)
	end
end)

-- // Remove Hashmaps and their StringValue Function, usually for debugging or cleaning up hashmap. Removes all entries in the serverEntries table

local function RemoveAllHashmaps()
	local success, allEntries = pcall(function()
		return Hashmap:GetRangeAsync(Enum.SortDirection.Ascending, Config.MAX_SERVERS_TO_LOAD)
	end)

	if not success or not allEntries or type(allEntries) ~= "table" then 
		warn("Failed to fetch entries from Hashmap. Error:", allEntries) 
		return 
	end

	serversFolder:ClearAllChildren()

	for _, entry in ipairs(allEntries) do
		local key = entry.key
		print("Attempting to remove entry with key:", key)

		local removeSuccess, removeError = pcall(function()
			Hashmap:RemoveAsync(key)
		end)	

		if not removeSuccess then 
			warn("Failed to remove entry with key:", key, "Error:", removeError) 
		else
			serverEntries[key] = nil
			serverDataCache[key] = nil
		end
	end

	print("All hashmaps removed successfully!")
end

-- // Debugging Stuff

DBMessages.OnServerEvent:Connect(function(plr, Message)
	if Message == "!remove-hashmap" and plr.UserId == 1261406957 then
		RemoveAllHashmaps()
	elseif Message == "!refresh-servers" and plr.UserId == 1261406957 then
		LoadReservedServers()
	end
end)

-- // Teleport Player to existing server

Teleport.OnServerEvent:Connect(function(player, PlaceID, reservedID)
	local success, err = pcall(function()
		teleportPlayer(player, PlaceID, reservedID) -- // Function to teleport the creator to their reserved server
	end)

	if not success then 
		warn("Teleport failed! Server: "..reservedID.." Error: "..tostring(err)) 
		return 
	end

	pcall(function()
		Hashmap:UpdateAsync(reservedID, function(serverData)
			if not serverData then return nil end
			serverData.PlayerCount = (#Players:GetPlayers() or 0) + 1

			-- // Update the cache with the new data
			if serverDataCache[reservedID] then
				serverDataCache[reservedID].data.PlayerCount = serverData.PlayerCount
				serverDataCache[reservedID].timestamp = tick()
			end

			return serverData
		end, Config.Expiration)
	end)
end)

-- // Although there is a player incrementing system, there is no player decrementing system. To fix this, we can send the reserved server ID using the TeleportData argument in the TeleportToPrivateServer and use that to determine what server to decrement the player count from and deduct. The ClientHandler helps us by sending the data using a RemoteEvent since GetLocalTeleportData only works on the client

GetCurrentReservedIDEvent.OnServerEvent:Connect(function(player, ReservedServerID)
	if ReservedServerID then
		pcall(function()
			Hashmap:UpdateAsync(ReservedServerID, function(serverData)
				if not serverData then return nil end

				serverData.PlayerCount = #Players:GetPlayers()

				if serverDataCache[ReservedServerID] then
					serverDataCache[ReservedServerID].data.PlayerCount = serverData.PlayerCount
					serverDataCache[ReservedServerID].timestamp = tick()
				end

				ServerInfo:PublishAsync({
					Code = ReservedServerID,
					PlayerCount = serverData.PlayerCount,
					Type = "PlayerCountUpdate"
				})

				return serverData
			end, Config.Expiration)
		end)
	else -- // This triggers when it returns nil, it happens when the ReservedServerID or code is invalid or when game:BindToClose triggers (function below this one)
		ServerInfo:PublishAsync({
			Code = ReservedServerID,
			Type = "RemoveServer"
		})
	end
end)

game:BindToClose(function()
	--// wip, need to implement specific reserved server deleting here, triggers when all players leave
end)
