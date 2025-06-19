local MatchmakingHandler = {}

-- // SERVICES

local MemoryStoreService = game:GetService("MemoryStoreService")
local MessagingService = game:GetService("MessagingService")
local HttpService = game:GetService("HttpService")
local TeleportService = game:GetService("TeleportService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")

-- // TRACKERS

local playerQueueKeys = {} -- // Stores the timestamp for each player based on their UserId and their modes
local localQueueCounts = {} -- // Tracks the amount of players in each queue. Stops all MemoryStoreService calls if there's noone in a certain subMode's mode
local isProcessingMatch = {} -- // Stores booleans for each player

local queueCache = {} -- // Stores Match Queue Maps
local messagingConnections = {} -- // All of the connections used from MessagingService

-- // MODULES

local Settings = require(script.Parent.Settings) -- // Handles all of the core information like modes, subModes, etc

-- // MISC

local lastCacheCleanup = 0 -- // Tracks how long until the next Cache cleanup
local heartbeatConnection

-- // UTILITY

local function getQueue(mode: string, subMode: string)
	local queueKey = mode .. "_" .. subMode
	if not queueCache[queueKey] then
		queueCache[queueKey] = {
			queue = MemoryStoreService:GetSortedMap("MatchQueue_" .. queueKey),
			lastUsed = tick()
		}
	else
		queueCache[queueKey].lastUsed = tick()
	end
	return queueCache[queueKey].queue
end

local function getLockMap()
	if not MatchmakingHandler._lockMap then
		MatchmakingHandler._lockMap = MemoryStoreService:GetSortedMap("MatchLocks")
	end
	return MatchmakingHandler._lockMap
end

local function getMatchSize(subMode: string)
	if Settings.MODE_CONFIG[subMode] then
		return Settings.MODE_CONFIG[subMode].totalPlayers
	end
	warn("Unknown subMode:", subMode, "- using default size of 2")
	return 2
end

local function cleanupCaches()
	local now = tick()

	for queueKey, cacheData in pairs(queueCache) do
		if now - cacheData.lastUsed > Settings.Matches.CACHE_CLEANUP_INTERVAL then
			queueCache[queueKey] = nil
		end	
	end

	if MatchmakingHandler.crossServerData then
		for queueKey, queueData in pairs(MatchmakingHandler.crossServerData) do
			for serverId, serverData in pairs(queueData) do
				if now - serverData.Timestamp > Settings.Matches.CROSS_SERVER_DATA_TTL then
					queueData[serverId] = nil
				end
			end

			if next(queueData) == nil then
				MatchmakingHandler.crossServerData[queueKey] = nil
			end
		end
	end

	lastCacheCleanup = now
end

function generatePlayerQueueData(player: Player, ELO: number, continent: string, subMode: string): string
	local timestamp = math.floor(tick() * 1000)
	local key = string.format("%010d", timestamp) .. "_" .. tostring(player.UserId)
	local data = {
		UserId = player.UserId,
		ELO = ELO,
		JoinTime = timestamp,
		ServerId = Settings.Information.SERVER_ID,
		Continent = continent,
		SubMode = subMode
	}
	return timestamp, key, HttpService:JSONEncode(data)
end

-- // MESSAGING SERVICE HANDLERS

local function setupMessagingService()
	for _, connection in pairs(messagingConnections) do
		connection:Disconnect()
	end

	messagingConnections = {}

	local success, connection = pcall(function()
		return MessagingService:SubscribeAsync("MatchmakingRequest", function(message) -- // Recieves the status of the current Matchmaking to all servers from the data of RequestCrossServerMatchmaking
			local data = message.Data
			if data.Mode and data.SubMode and data.RequestingServer ~= Settings.Information.SERVER_ID then
				task.spawn(function()
					MatchmakingHandler:SendQueueStatus(data.Mode, data.SubMode, data.RequestingServer)
				end)
			end
		end)
	end)

	if success then
		messagingConnections.matchmakingRequest = connection
	end

	success, connection = pcall(function() -- // Teleports all affected players to 1 Private Servers
		return MessagingService:SubscribeAsync("CrossServerTeleport", function(message)
			local data = message.Data
			if data.PlayersForServer and data.PlayersForServer[Settings.Information.SERVER_ID] then
				task.spawn(function()
					MatchmakingHandler:HandleTeleportRequest(data)
				end)
			end
		end)
	end)

	if success then
		messagingConnections.crossServerTeleport = connection
	end

	success, connection = pcall(function()
		return MessagingService:SubscribeAsync("QueueStatusResponse", function(message)
			local data = message.Data
			if data.TargetServer == Settings.Information.SERVER_ID then
				MatchmakingHandler:ReceiveQueueStatus(data)
			end
		end)
	end)

	if success then
		messagingConnections.queueStatusResponse = connection
	end
end

function MatchmakingHandler:RequestCrossServerMatchmaking(mode: string, subMode: string)
	pcall(function()
		MessagingService:PublishAsync("MatchmakingRequest", {
			Mode = mode,
			SubMode = subMode,
			RequestingServer = Settings.Information.SERVER_ID,
			Timestamp = tick()
		})
	end)
end

function MatchmakingHandler:SendQueueStatus(mode: string, subMode: string, targetServer: string) -- // Sends all of the information in the MemoryStore sorted map and broadcasts it to  QueueStatusResponse
	local queue = getQueue(mode, subMode)
	local players = {}
	local matchSize = getMatchSize(subMode)
	local maxPlayers = math.min(50, matchSize * 3)

	local success, result = pcall(function()
		return queue:GetRangeAsync(Enum.SortDirection.Ascending, maxPlayers)
	end)

	if success and result then
		for _, entry in ipairs(result) do
			local success, data = pcall(function()
				return HttpService:JSONDecode(entry.value)
			end)
			if success and data and data.ServerId == Settings.Information.SERVER_ID then
				local player = Players:GetPlayerByUserId(data.UserId)
				if player then
					table.insert(players, {
						UserId = data.UserId,
						ELO = data.ELO,
						JoinTime = data.JoinTime,
						QueueKey = entry.key
					})
				end
			end
		end
	end

	pcall(function()
		MessagingService:PublishAsync("QueueStatusResponse", {
			Mode = mode,
			SubMode = subMode,
			TargetServer = targetServer,
			SendingServer = Settings.Information.SERVER_ID,
			Players = players,
			PlayerCount = #players,
			Timestamp = tick()
		})
	end)
end

function MatchmakingHandler:ReceiveQueueStatus(data) -- // Stores the Cross Server data for MessagingService to use and broadcast to servers
	if not self.crossServerData then
		self.crossServerData = {}
	end

	local queueKey = data.Mode .. "_" .. data.SubMode

	if not self.crossServerData[queueKey] then
		self.crossServerData[queueKey] = {}
	end

	self.crossServerData[queueKey][data.SendingServer] = {
		Players = data.Players,
		Timestamp = data.Timestamp
	}

	local now = tick()
	for serverId, serverData in pairs(self.crossServerData[queueKey]) do
		if now - serverData.Timestamp > Settings.Matches.CROSS_SERVER_DATA_TTL then
			self.crossServerData[queueKey][serverId] = nil
		end
	end
end

-- // MATCH-MAKING

function MatchmakingHandler:CanStartMatchmaking(queueKey: string, subMode: string): boolean -- // Just some basic checks before getting into real matchmaking
	if isProcessingMatch[queueKey] then
		return false
	end

	local matchSize = getMatchSize(subMode)

	if matchSize % 2 ~= 0 then
		warn("Invalid match size for", subMode, "- must be even number")
		return false
	end

	return true
end

function MatchmakingHandler:ProcessMatchmaking(mode: string, subMode: string, queueKey: string)
	local allPlayers = self:CollectPlayers(mode, subMode, queueKey)

	if not allPlayers or #allPlayers == 0 or #allPlayers < getMatchSize(subMode) then
		return
	end
	
	self:RequestCrossServerMatchmaking(mode, subMode)

	local bestGroup = self:FindOptimalPlayerGroup(allPlayers, subMode)

	if bestGroup then
		self:CreateCrossServerMatch(bestGroup, mode, subMode)
	else
		warn("No optimal group found for", mode, subMode)
	end
end

function MatchmakingHandler:CollectPlayers(mode: string, subMode: string, queueKey: string): {any}
	if not localQueueCounts[queueKey] or localQueueCounts[queueKey] == 0 then
		return {}
	end

	local allPlayers = {}
	local queue = getQueue(mode, subMode)
	local batchSize = math.min(200, getMatchSize(subMode) * Settings.Matches.BATCH_SIZE)

	print("Processing matchmaking: Found", localQueueCounts[queueKey], "total players for", mode, subMode)

	local success, result = pcall(function()
		return queue:GetRangeAsync(Enum.SortDirection.Ascending, batchSize)
	end)

	if success and type(result) == "table" then
		for _, entry in ipairs(result) do
			local playerData = self:ParseQueueEntry(entry)
			if playerData then
				table.insert(allPlayers, playerData)
			end
		end
	else
		warn("Failed to collect players from queue:", queueKey)
	end

	return allPlayers
end

function MatchmakingHandler:ParseQueueEntry(entry: any)
	local success, data = pcall(function()
		return HttpService:JSONDecode(entry.value)
	end)

	if not success or not data or not data.UserId or not data.ELO then
		return nil
	end

	return {
		UserId = data.UserId,
		ELO = data.ELO,
		JoinTime = data.JoinTime,
		QueueKey = entry.key,
		ServerId = data.ServerId,
		Continent = data.Continent or "Unknown"
	}
end

function MatchmakingHandler:FindOptimalPlayerGroup(allPlayers: {any}, subMode: string): {any}?
	local matchSize = getMatchSize(subMode)
	local continentGroups = self:GroupPlayersByContinent(allPlayers)

	self:SortPlayers(allPlayers)

	for continent, playersInContinent in pairs(continentGroups) do
		if #playersInContinent >= matchSize then
			local bestGroup = self:FindBestGroup(playersInContinent, matchSize)
			if bestGroup then
				return bestGroup
			end
		end
	end

	return self:FindBestGroup(allPlayers, matchSize)
end

function MatchmakingHandler:GroupPlayersByContinent(players: {any}): {[string]: {any}}
	local groups = {}
	for _, player in ipairs(players) do
		local cont = player.Continent or "Unknown"
		groups[cont] = groups[cont] or {}
		table.insert(groups[cont], player)
	end
	return groups
end

function MatchmakingHandler:SortPlayers(players: {any})
	table.sort(players, function(a, b)
		if Settings.Matches.PRIORITIZE_WAIT_TIME and a.JoinTime ~= b.JoinTime then
			return a.JoinTime < b.JoinTime
		end
		return a.ELO < b.ELO
	end)
end

function MatchmakingHandler:FindBestGroup(players: {any}, matchSize: number)
	if not players or #players < matchSize then
		return nil
	end

	local bestGroup = nil
	local bestSpread = math.huge
	local maxIterations = math.min(#players - matchSize + 1, 50)
	local currentTime = tick()

	for i = 1, maxIterations do
		local group = {}
		for j = i, i + matchSize - 1 do
			table.insert(group, players[j])
		end

		local longestWait = 0
		for _, player in pairs(group) do
			local waitTime = currentTime - player.JoinTime
			longestWait = math.max(longestWait, waitTime)
		end

		local baseSpread = Settings.Matches.ELO_CHANGE or 200
		local increaseRate = Settings.Matches.ELO_INCREASE_RATE or 5
		local allowedSpread = baseSpread + (longestWait * increaseRate)

		local minELO = group[1].ELO
		local maxELO = group[#group].ELO
		local actualSpread = maxELO - minELO

		if actualSpread <= allowedSpread and actualSpread < bestSpread then
			bestSpread = actualSpread
			bestGroup = group
			if actualSpread == 0 then
				break
			end
		end
	end

	return bestGroup
end

function MatchmakingHandler:CreateCrossServerMatch(players: {any}, mode: string, subMode: string) -- // Notifies all of the affected servers to Match make players all notifies servers to teleport them
	if not players or #players == 0 then
		warn("No players provided for match creation")
		return false
	end

	print("Creating cross-server match with", #players, "players for", mode, subMode)

	local lockMap = getLockMap()
	local lockedPlayerIds = {}
	local allPlayersValid = true

	for _, playerData in ipairs(players) do
		if not playerData.UserId then
			warn("Invalid player data missing UserId")
			allPlayersValid = false
			break
		end

		local success, err = pcall(function()
			return lockMap:UpdateAsync(tostring(playerData.UserId), {
				Status = "InMatch",
				MatchId = Settings.Information.SERVER_ID,
				Timestamp = tick()
			}, Settings.Matches.LOCK_TIMEOUT)
		end)

		if not success then
			warn("Failed to lock player", playerData.UserId, ":", err)
			allPlayersValid = false
			break
		end

		table.insert(lockedPlayerIds, playerData.UserId)
	end

	if not allPlayersValid then
		self:UnlockPlayers(lockedPlayerIds)
		return false
	end

	local code, reserveErr
	local success, err = pcall(function()
		code, reserveErr = TeleportService:ReserveServer(Settings.Information.PLACE_ID)
	end)

	if not success or not code then
		warn("Failed to reserve server:", err or reserveErr)
		self:UnlockPlayers(lockedPlayerIds)
		return false
	end

	local playersByServer = {}
	local removalFailures = 0
	local queue = getQueue(mode, subMode)

	for _, playerData in ipairs(players) do
		local removeSuccess = pcall(function()
			queue:RemoveAsync(playerData.QueueKey)
		end)

		if not removeSuccess then
			removalFailures += 1
			warn("Failed to remove player from queue:", playerData.UserId)
		end

		if playerData.ServerId == Settings.Information.SERVER_ID then
			local player = Players:GetPlayerByUserId(playerData.UserId)
			if player and playerQueueKeys[playerData.UserId] then
				playerQueueKeys[playerData.UserId][mode.."_"..subMode] = nil
				if next(playerQueueKeys[playerData.UserId]) == nil then
					playerQueueKeys[playerData.UserId] = nil
				end
			end
		end

		local serverId = playerData.ServerId
		playersByServer[serverId] = playersByServer[serverId] or {}
		table.insert(playersByServer[serverId], playerData.UserId)
	end

	local queueKey = mode .. "_" .. subMode
	local localPlayersRemoved = 0

	for _, playerData in ipairs(players) do
		if playerData.ServerId == Settings.Information.SERVER_ID then
			localPlayersRemoved += 1
		end
	end

	localQueueCounts[queueKey] = math.max(0, (localQueueCounts[queueKey] or 0) - localPlayersRemoved)

	if removalFailures > 0 then
		warn(removalFailures, "queue removal failures occurred")
	end

	local notifySuccess, notifyErr = pcall(function()
		MessagingService:PublishAsync("CrossServerTeleport", {
			MatchId = Settings.Information.SERVER_ID,
			ServerCode = code,
			PlaceId = Settings.Information.PLACE_ID,
			PlayersForServer = playersByServer,
			Mode = mode,
			SubMode = subMode,
			Timestamp = tick()
		})
	end)

	if not notifySuccess then
		warn("Failed to notify servers:", notifyErr)
		self:UnlockPlayers(lockedPlayerIds)
		return false
	end

	print("Successfully created cross-server match", Settings.Information.SERVER_ID)
	return true
end

function MatchmakingHandler:UnlockPlayers(playerIds: {number}) -- // Removes players from the lockMap MemoryStore
	local lockMap = getLockMap()
	for _, userId in ipairs(playerIds) do
		pcall(function()
			lockMap:RemoveAsync(tostring(userId))
		end)
	end
end

function MatchmakingHandler:HandleTeleportRequest(data)
	local playersToTeleport = {}
	local playerIds = data.PlayersForServer[Settings.Information.SERVER_ID]
	if not playerIds then return end

	for _, userId in ipairs(playerIds) do
		local player = Players:GetPlayerByUserId(userId)
		if player then
			table.insert(playersToTeleport, player)
		end
	end

	if #playersToTeleport == 0 then return end

	print("Teleporting", #playersToTeleport, "players to cross-server match")

	local playersToRetry = playersToTeleport
	local maxRetries = Settings.Matches.TELEPORT_RETRIES

	for attempt = 1, maxRetries do
		if #playersToRetry == 0 then break end

		local success, err = pcall(function()
			TeleportService:TeleportToPrivateServer(
				data.PlaceId,
				data.ServerCode,
				playersToRetry,
				nil,
				{
					MatchId = data.MatchId,
					Mode = data.Mode,
					SubMode = data.SubMode
				}
			)
		end)

		if success then
			print(`Successfully teleported {#playersToRetry} players on attempt {attempt}`)
			break
		else
			local stillValidPlayers = {}
			for _, player in ipairs(playersToRetry) do
				if player.Parent and Players:GetPlayerByUserId(player.UserId) then
					table.insert(stillValidPlayers, player)
				end
			end
			playersToRetry = stillValidPlayers

			if attempt < maxRetries and #playersToRetry > 0 then
				task.wait(0.5 + (attempt * 0.2)) -- // Linear backoff
			end
		end
	end

	if #playersToRetry > 0 then
		warn(`Failed to teleport {#playersToRetry} players after {maxRetries} attempts`)
		local playerIds = {}
		for _, player in ipairs(playersToRetry) do
			table.insert(playerIds, player.UserId)
		end
		self:UnlockPlayers(playerIds)
	end
end

-- // MAIN FUNCTIONS

-- QueuePlayer(): Queues the player(s) up into the matchmaking
-- @param playerOrList: The player(s) to queue. Put a list of players to bulk queue them or just add one player
-- @param ELO: The amount of ELO the player has
-- @param mode: The mode the player should be queued to (Settings.Modes)
-- @param playerContinent: The continent the player is in (Handled by the PlayerLocation) module
-- @param subMode: The Sub-Mode the player should be queued to (Keys of Settings.MODE_CONFIG)
function MatchmakingHandler:QueuePlayer(playerOrList: Player | {Player}, ELO: number, mode: string, playerContinent: string, subMode: string)
	local queueKey = mode .. "_" .. subMode
	local players = typeof(playerOrList) == "Instance" and { playerOrList } or playerOrList

	for _, player in players do
		if playerQueueKeys[player.UserId] and playerQueueKeys[player.UserId][queueKey] then
			warn(player.Name .. " is already in the queue for mode:", mode, subMode)
			continue
		end
		
		local timestamp, key, data = generatePlayerQueueData(playerOrList, ELO, playerContinent, subMode)

		if not playerQueueKeys[player.UserId] then
			playerQueueKeys[player.UserId] = {}
		end
		playerQueueKeys[player.UserId][queueKey] = key

		local success, err = pcall(function()
			getQueue(mode, subMode):SetAsync(key, data, Settings.Matches.MATCHMAKING_DURATION)
		end)

		if not success then
			warn("Failed to queue player:", err)
			playerQueueKeys[player.UserId][queueKey] = nil
		else
			localQueueCounts[queueKey] = math.max(0, (localQueueCounts[queueKey] or 0) + 1)
			print("Queued player", player.Name, "for", mode, subMode, "mode. Region: " .. playerContinent)
		end
	end
end

-- Stops the player(s) queue for every Mode and subMode
-- @param playerOrList: The player(s) to stop queuing. Pass a list to bulk remove
function MatchmakingHandler:StopQueue(playerOrList: Player | {Player}): boolean
	local players = typeof(playerOrList) == "Instance" and { playerOrList } or playerOrList
	local allSuccess = true
	local lockMap = getLockMap()

	for _, player in players do
		if not player or not player.UserId then
			allSuccess = false
			continue
		end

		local playerQueues = playerQueueKeys[player.UserId]
		if playerQueues then
			for queueKey, keyToRemove in pairs(playerQueues) do
				local mode, subMode = queueKey:match("(.+)_(.+)")
				if mode and subMode then
					local success = pcall(function()
						getQueue(mode, subMode):RemoveAsync(keyToRemove)
					end)
					if success then
						localQueueCounts[queueKey] = math.max(0, (localQueueCounts[queueKey] or 0) - 1)
					else
						warn("Failed to remove from queue:", queueKey)
						allSuccess = false
					end
				end
			end
		end

		pcall(function()
			lockMap:RemoveAsync(tostring(player.UserId))
		end)

		playerQueueKeys[player.UserId] = nil
	end

	return allSuccess
end

-- MatchMake(): Globally handles and checks all players elligble for Matchmaking based on their Mode and subMode and teleports them
-- @param mode: The mode the player should be queued to (Settings.Modes)
-- @param subMode: The Sub-Mode the player should be queued to (Keys of Settings.MODE_CONFIG)
function MatchmakingHandler:MatchMake(mode: string, subMode: string)
	local queueKey = mode .. "_" .. subMode

	if not self:CanStartMatchmaking(queueKey, subMode) then
		return
	end

	isProcessingMatch[queueKey] = true

	local success, err = pcall(function()
		self:ProcessMatchmaking(mode, subMode, queueKey)
	end)

	isProcessingMatch[queueKey] = false

	if not success then
		warn("Error in CrossServerMatchMake:", err, debug.traceback())
	end
end

-- // MISC FUNCTIONS

-- Cleanup(): Cleans up all caches
function MatchmakingHandler:Cleanup()
	for _, connection in pairs(messagingConnections) do
		pcall(function() connection:Disconnect() end)
	end

	messagingConnections = {}

	queueCache = {}

	if self.crossServerData then
		self.crossServerData = {}
	end
 
	playerQueueKeys = {}

	if heartbeatConnection then
		heartbeatConnection:Disconnect()
	end

	isProcessingMatch = {}
end

-- IsPlayerQueued(): Checks if a player is queued in a certain subMode's mode
-- @param player: The player to check
-- @param mode: The mode the player should be queued to (Settings.Modes)
-- @param subMode: The Sub-Mode the player should be queued to (Keys of Settings.MODE_CONFIG)
function MatchmakingHandler:IsPlayerQueued(player: Player, mode: string?, subMode: string?): boolean
	local queues = playerQueueKeys[player.UserId]
	if not queues then return false end

	if mode and subMode then
		local key = mode .. "_" .. subMode
		return queues[key] ~= nil
	end

	return next(queues) ~= nil
end

-- GetQueuedPlayers(): Get all of the queued players in a subMode's mode
-- @param mode: The mode the player should be queued to (Settings.Modes)
-- @param subMode: The Sub-Mode the player should be queued to (Keys of Settings.MODE_CONFIG)
function MatchmakingHandler:GetQueuedPlayers(mode: string, subMode: string): {number}
	local queue = getQueue(mode, subMode)
	local players = {}
	local batchSize = math.min(200, getMatchSize(subMode) * Settings.Matches.BATCH_SIZE)
	local success, result = pcall(function()
		return queue:GetRangeAsync(Enum.SortDirection.Ascending, batchSize)
	end)

	if success and result then
		for _, entry in result do
			local data = HttpService:JSONDecode(entry.Value)
			table.insert(players, data.UserId)
		end
	end

	return players
end

-- GetPlayerJoinTime(): Gets the players join time in a subMode's mode
-- @param player: The player to check
-- @param mode: The mode the player should be queued to (Settings.Modes)
-- @param subMode: The Sub-Mode the player should be queued to (Keys of Settings.MODE_CONFIG)
function MatchmakingHandler:GetPlayerJoinTime(player: Player, mode: string, subMode: string): number?
	local queueKey = mode .. "_" .. subMode
	local key = playerQueueKeys[player.UserId] and playerQueueKeys[player.UserId][queueKey]
	if not key then return nil end

	local queue = getQueue(mode, subMode)
	local success, value = pcall(function()
		return queue:GetAsync(key)
	end)

	if success and value then
		local data = HttpService:JSONDecode(value)
		return data.JoinTime
	end

	return nil
end

-- GetQueueCount(): Gets the local queue count in a subMode's mode
-- @param mode: The mode the player should be queued to (Settings.Modes)
-- @param subMode: The Sub-Mode the player should be queued to (Keys of Settings.MODE_CONFIG)
function MatchmakingHandler:GetQueueCount(mode: string, subMode: string): number
	local key = mode .. "_" .. subMode
	return localQueueCounts[key] or 0
end

-- // INITIALIZATION

setupMessagingService()

Players.PlayerRemoving:Connect(function(player)
	if player and player.UserId then
		MatchmakingHandler:StopQueue(player)
	end
end)

game:BindToClose(function()
	MatchmakingHandler:Cleanup()
end)

-- // LOOPS

heartbeatConnection = RunService.Heartbeat:Connect(function()
	if tick() - lastCacheCleanup > Settings.Matches.CACHE_CLEANUP_INTERVAL then
		cleanupCaches()
	end
end)

return MatchmakingHandler
