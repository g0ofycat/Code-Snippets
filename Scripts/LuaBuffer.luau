local LuaBuffer = {}
LuaBuffer.__index = LuaBuffer

---------------------------------------------
-- // Constructor
---------------------------------------------

function LuaBuffer.New(initialSize: number?)
	local self = setmetatable({
		buffer = table.create(initialSize or 32), -- // The reason we are using table.create() instead of {} is to avoid rehashing (The resizing of tables). When inserting a value into a table, it may expand causing performance issues
		bitPosition = 0, -- // Tracks the current bit position within the buffer, when it reaches a multiple of 8, the byte is stored
		bytePosition = 1, -- // The current position of the byte, allows us to see if the next byte should be inserted into the buffer by checking if it's too large to hold the new byte. If not then just overwrite the existing byte at bytePosition
		currentByte = 0, -- // The actual byte we're on in WriteBit. Used to store and overwrite specific bytes in the buffer table
		totalBitsWritten = 0, -- // Tracks total bits written to limit reads to valid data. (Which are fully written bytes)
		readBitPosition = 0 -- // Tracks read position separately from write position
	}, LuaBuffer)
	return self
end

---------------------------------------------
-- // Reading and Writing Bits (MSB - LSB)
---------------------------------------------

function LuaBuffer:WriteBit(bit: number)
	if bit ~= 0 and bit ~= 1 then
		error("Bit must be 0 or 1, got " .. tostring(bit))
	end
	local shiftAmount = self.bitPosition % 8
	local shiftedBit = bit32.lshift(bit, shiftAmount)
	self.currentByte = bit32.bor(self.currentByte, shiftedBit)
	self.bitPosition = self.bitPosition + 1
	self.totalBitsWritten = self.totalBitsWritten + 1

	-- // If 8 bits accumulated, store the byte
	if self.bitPosition % 8 == 0 then
		local buffer = self.buffer
		local bytePos = math.floor(self.bitPosition / 8) -- // Align bytePos with completed bytes
		local curByte = self.currentByte

		if bytePos > #buffer then
			table.insert(buffer, curByte) -- // Insert new byte
		else
			buffer[bytePos] = curByte -- // Overwrite only if within bounds
		end

		self.bytePosition = bytePos + 1 -- // Next available byte position
		self.currentByte = 0
	end
end

function LuaBuffer:WriteBits(value: number, bitCount: number)
	if not (value >= 0 and bitCount > 0 and bitCount <= 52) then -- // Filter out negative numbers and numbers over 52. The reason for this is because of how Luau handles double-precision floating-point numbers it lose integer precision beyond 52-53 bits.
		error("bitCount must be positive and not exceed 52")
	end

	local maxValue = 2^bitCount
	if value >= maxValue then
		error("Value out of range for " .. bitCount .. " bits. Use a higher bitCount.")
	end

	local remainingBits = bitCount
	local buffer = self.buffer

	-- // Handle pending bits in currentByte first
	if self.bitPosition % 8 ~= 0 then
		for i = bitCount - 1, 0, -1 do
			local bit = bit32.band(bit32.rshift(value, i), 1)
			self:WriteBit(bit)
		end
		return
	end

	-- // Read whole bytes when possible (Faster)
	while remainingBits >= 8 do
		local byte = bit32.band(bit32.rshift(value, remainingBits - 8), 0xFF)

		if self.bytePosition > #buffer then
			table.insert(buffer, byte)
		else
			buffer[self.bytePosition] = byte
		end

		self.bytePosition = self.bytePosition + 1
		self.bitPosition = self.bitPosition + 8
		self.totalBitsWritten = self.totalBitsWritten + 8
		remainingBits = remainingBits - 8
	end

	-- // Read remaining bits (If any)
	if remainingBits > 0 then
		local mask = bit32.lshift(1, remainingBits) - 1
		local remainingValue = bit32.band(value, mask)

		for i = remainingBits - 1, 0, -1 do
			local bit = bit32.band(bit32.rshift(remainingValue, i), 1)
			self:WriteBit(bit)
		end
	end
end

function LuaBuffer:ReadBit()
	local byteIndex = bit32.rshift(self.readBitPosition, 3) + 1

	if byteIndex > #self.buffer then
		error("Attempted to read beyond buffer length")
	end
	
	if self.readBitPosition >= self.totalBitsWritten then
		error("Attempted to read beyond written bits")
	end

	local byte = self.buffer[byteIndex]
	local bitIndex = self.readBitPosition % 8
	local bit = bit32.band(bit32.rshift(byte, bitIndex), 1)

	self.readBitPosition = self.readBitPosition + 1
	return bit
end

function LuaBuffer:ReadBits(bitCount)
	if not (bitCount > 0 and bitCount <= 52) then -- // Filter out negative numbers and numbers over 52. The reason for this is because of how Luau handles double-precision floating-point numbers it lose integer precision beyond 52-53 bits.
		error("bitCount must be between 1 and 52")
	end
	
	self:Flush()

	local availableBits = self.totalBitsWritten - self.readBitPosition

	if bitCount > availableBits then
		error(string.format("Requested %d bits, only %d available", bitCount, availableBits))
	end

	local value = 0
	local remainingBits = bitCount
	local buffer = self.buffer

	-- // Read whole bytes when possible (Faster)
	while remainingBits >= 8 and (self.readBitPosition % 8) == 0 do
		local byteIndex = bit32.rshift(self.readBitPosition, 3) + 1
		local byte = buffer[byteIndex]

		value = bit32.bor(bit32.lshift(value, 8), byte)

		self.readBitPosition = self.readBitPosition + 8
		remainingBits = remainingBits - 8
	end

	-- // Read remaining bits (If any)
	while remainingBits > 0 do
		local byteIndex = bit32.rshift(self.readBitPosition, 3) + 1
		local byte = buffer[byteIndex]
		local bitPosInByte = self.readBitPosition % 8
		local bit = bit32.band(bit32.rshift(byte, bitPosInByte), 1)

		value = bit32.bor(bit32.lshift(value, 1), bit)
		self.readBitPosition = self.readBitPosition + 1
		remainingBits = remainingBits - 1
	end

	return value
end

---------------------------------------------
-- // Utility Methods
---------------------------------------------

function LuaBuffer:Reset() -- // We need to actually reset the positions and the current byte were on since WriteBit() doesn't actually immediately store each byte in the self.buffer table, it only inserts it into the table when theres a full byte (8 bits).
	self.bitPosition = 0
	self.bytePosition = 1
	self.currentByte = 0
	self.readBitPosition = 0
end	

function LuaBuffer:Flush()
	if self.bitPosition % 8 ~= 0 then -- // If bitPosition isn't a byte boundary
		local validBits = self.bitPosition % 8
		local mask = bit32.lshift(1, validBits) - 1 -- // Only valid bits (e.g., 3 bits -> 00000111)

		self.currentByte = bit32.band(self.currentByte, mask) -- // Clear unwritten bits

		local buffer = self.buffer
		local bytePos = math.floor(self.bitPosition / 8) + 1 -- // Position for the partial byte
		local curByte = self.currentByte

		if bytePos > #buffer then
			table.insert(buffer, curByte)
		else
			buffer[bytePos] = curByte
		end

		self.bytePosition = bytePos + 1
		self.currentByte = 0
	end
end

---------------------------------------------
-- // Data Types
---------------------------------------------

return LuaBuffer
