--[[ README:
	A bit buffer module meant for optomizations. Just a personal project that was fun to create;
	You might be better off using other Bit Buffer modules like BitBuf.
--]]

local LuaBuffer = {}
LuaBuffer.__index = LuaBuffer

---------------------------------------------
-- // Constructor
---------------------------------------------

function LuaBuffer.New(initialSize: number?)
	initialSize = initialSize or 32
	if initialSize <= 0 then
		error("initialSize must be positive")
	end
	local self = setmetatable({
		buffer = table.create(initialSize), -- // The reason we are using table.create() instead of {} is to avoid rehashing (The resizing of tables). When inserting a value into a table, it may expand causing performance issues
		bufferSize = initialSize, -- // Tracks buffer capacity, used in CheckSpace to see if we can write in buffer
		bitPosition = 0, -- // Tracks the current bit position within the buffer, when it reaches a multiple of 8, the byte is stored
		bytePosition = 1, -- // The current position of the byte, allows us to see if the next byte should be inserted into the buffer by checking if it's too large to hold the new byte. If not then just overwrite the existing byte at bytePosition
		currentByte = 0, -- // The actual byte we're on in WriteBit. Used to store and overwrite specific bytes in the buffer table
		totalBitsWritten = 0, -- // Tracks total bits written to limit reads to valid data. (Which are fully written bytes)
		readBitPosition = 0 -- // Tracks read position separately from write position
	}, LuaBuffer)
	return self
end

---------------------------------------------
-- // Helper Methods
---------------------------------------------

function LuaBuffer:CheckSpace(neededCapacity: number) -- // Used to see if there's actually enough space in the buffer before writing. Creates a new table with more space and all of the contents of the old table
	if neededCapacity <= self.bufferSize then return end -- // No need to compute anything if theres already space
	-- // Geometric growth (1.5x) to balance memory usage and performance
	local newSize = math.max(math.floor(self.bufferSize * 1.5), neededCapacity + 1)
	local newBuffer = table.create(newSize)
	-- // Copy existing data to newBuffer
	for i = 1, #self.buffer do
		newBuffer[i] = self.buffer[i]
	end
	
	self.buffer = newBuffer
	self.bufferSize = newSize
end

---------------------------------------------
-- // Reading and Writing Bits (MSB - LSB)
---------------------------------------------

function LuaBuffer:WriteBit(bit: number)
	if bit ~= 0 and bit ~= 1 then
		error(string.format("Bit must be 0 or 1, got %d", bit))
	end
	local shiftAmount = 7 - (self.bitPosition % 8)
	local shiftedBit = bit32.lshift(bit, shiftAmount)
	self.currentByte = bit32.bor(self.currentByte, shiftedBit)
	self.bitPosition = self.bitPosition + 1
	self.totalBitsWritten = self.totalBitsWritten + 1
	if self.bitPosition % 8 == 0 then -- // If 8 bits, store the byte
		local bytePos = math.floor(self.bitPosition / 8)
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = self.currentByte
		self.bytePosition = bytePos + 1
		self.currentByte = 0
	end
end

function LuaBuffer:WriteBits(value: number, bitCount: number)
	if not (value >= 0 and bitCount > 0 and bitCount < 53) then -- // Filter out negative numbers and numbers over 52. The reason for this is because of how Luau handles double-precision floating-point numbers it lose integer precision beyond 52-53 bits
		error(string.format("bitCount must be 1-52 (received %d) due to Lua number precision limits", bitCount))
	end
	local maxValue = 2^bitCount
	if value >= maxValue then
		error(string.format("Value out of range for %d bits. Use a higher bitCount.", bitCount))
	end

	-- // Handle full bytes directly when aligned
	local bitsLeft = bitCount
	while bitsLeft >= 8 and self.bitPosition % 8 == 0 do
		local byteShift = bitsLeft - 8
		local byte = bit32.band(bit32.rshift(value, byteShift), 0xFF)
		-- // We can directly store byte to buffer to avoid bit-by-bit processing
		local bytePos = math.floor(self.bitPosition / 8) + 1
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = byte
		self.bitPosition = self.bitPosition + 8
		self.totalBitsWritten = self.totalBitsWritten + 8
		self.bytePosition = bytePos + 1
		bitsLeft = bitsLeft - 8
	end

	-- // Handle remaining bits
	for i = bitsLeft - 1, 0, -1 do
		local bit = bit32.band(bit32.rshift(value, i), 1)
		self:WriteBit(bit)
	end
end

function LuaBuffer:ReadBit()
    self:Flush()
    local byteIndex = bit32.rshift(self.readBitPosition, 3) + 1
    
    if self.readBitPosition >= self.totalBitsWritten then
        error("Attempted to read beyond written bits")
    end
    
    if byteIndex > #self.buffer or self.buffer[byteIndex] == nil then
        error("Attempted to read beyond buffer length or from nil byte at byteIndex " .. byteIndex)
    end

    local byte = self.buffer[byteIndex]
    local bitIndex = 7 - (self.readBitPosition % 8)
    local bit = bit32.band(bit32.rshift(byte, bitIndex), 1)
    self.readBitPosition = self.readBitPosition + 1
    return bit
end

function LuaBuffer:ReadBits(bitCount: number)
	self:Flush()	
	if not (bitCount > 0 and bitCount < 53) then
		error(string.format("bitCount must be 1-52 (received %d) due to Lua number precision limits", bitCount))
	end

	local availableBits = self.totalBitsWritten - self.readBitPosition
	if bitCount > availableBits then
		error(string.format("Requested %d bits, only %d available", bitCount, availableBits))
	end

	local value = 0
	local bitsLeft = bitCount

	-- // Read full bytes at once when aligned
	while bitsLeft >= 8 and self.readBitPosition % 8 == 0 do
		local byteIndex = bit32.rshift(self.readBitPosition, 3) + 1
		if byteIndex > #self.buffer then
			error("Attempted to read beyond buffer length at byteIndex " .. byteIndex)
		end
		value = bit32.lshift(value, 8) + self.buffer[byteIndex]
		self.readBitPosition = self.readBitPosition + 8
		bitsLeft = bitsLeft - 8
	end

	-- // Read remaining bits
	for i = 1, bitsLeft do
		value = bit32.lshift(value, 1) + self:ReadBit()
	end

	return value
end

---------------------------------------------
-- // Utility Methods
---------------------------------------------

function LuaBuffer:Reset()
	self.bitPosition = 0
	self.bytePosition = 1
	self.currentByte = 0
	self.totalBitsWritten = 0
	self.readBitPosition = 0
	table.clear(self.buffer)
end

function LuaBuffer:ResetReadPosition()
	self:Flush()
	self.readBitPosition = 0
end

function LuaBuffer:SeekReadPosition(bitPosition: number)
	self:Flush()
	if bitPosition < 0 or bitPosition > self.totalBitsWritten then
		error(string.format("Bit position %d out of range (0-%d)", 
			bitPosition, self.totalBitsWritten))
	end
	self.readBitPosition = bitPosition
end

function LuaBuffer:Flush() -- // Incomplete bytes are filled with 0's then added to the buffer, might wanna change this
	if self.bitPosition % 8 ~= 0 then
		local bytePos = math.floor(self.bitPosition / 8) + 1
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = self.currentByte

		self.totalBitsWritten = math.max(self.totalBitsWritten, self.bitPosition)
	end
end

function LuaBuffer:WriteByte(byte: number)
	if byte < 0 or byte > 255 then
		error(string.format("Byte must be between 0 and 255, got %d", byte))
	end

	-- // If we're byte-aligned, write directly
	if self.bitPosition % 8 == 0 then
		local bytePos = math.floor(self.bitPosition / 8) + 1
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = byte
		self.bitPosition = self.bitPosition + 8
		self.totalBitsWritten = self.totalBitsWritten + 8
		self.bytePosition = bytePos + 1
	else -- // Else? Write it by each bit
		for i = 7, 0, -1 do
			self:WriteBit(bit32.band(bit32.rshift(byte, i), 1))
		end
	end
end

function LuaBuffer:GetBuffer()
	self:Flush()
	return self.buffer
end

function LuaBuffer:GetWrittenBytes()
	self:Flush()
	local byteCount = math.ceil(self.totalBitsWritten / 8)
	local result = table.create(byteCount)

	for i = 1, byteCount do
		result[i] = self.buffer[i] or 0
	end

	-- // Mask last byte if we don't have complete 8 bits
	local extraBits = (8 - (self.totalBitsWritten % 8)) % 8
	if extraBits > 0 then
		local mask = bit32.lshift(1, extraBits) - 1
		mask = bit32.bnot(mask) -- // Invert to keep the valid bits
		result[byteCount] = bit32.band(result[byteCount], mask)
	end

	return result
end

function LuaBuffer:GetBitPosition()
	return self.bitPosition
end

function LuaBuffer:GetBytePosition()
	return self.bytePosition
end

---------------------------------------------
-- // Data Types
---------------------------------------------

return LuaBuffer
