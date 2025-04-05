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
	local shiftAmount = 7 - (self.bitPosition % 8)
	local shiftedBit = bit32.lshift(bit, shiftAmount)
	self.currentByte = bit32.bor(self.currentByte, shiftedBit)
	self.bitPosition = self.bitPosition + 1
	self.totalBitsWritten = self.totalBitsWritten + 1
	if self.bitPosition % 8 == 0 then -- // If 8 bits, store the byte
		local buffer = self.buffer
		local bytePos = math.floor(self.bitPosition / 8) 
		if bytePos > #buffer then
			table.insert(buffer, self.currentByte)
		else
			buffer[bytePos] = self.currentByte
		end
		self.bytePosition = bytePos + 1
		self.currentByte = 0
	end
end

function LuaBuffer:WriteBits(value: number, bitCount: number)
	if not (value >= 0 and bitCount > 0 and bitCount <= 52) then -- // Filter out negative numbers and numbers over 52. The reason for this is because of how Luau handles double-precision floating-point numbers it lose integer precision beyond 52-53 bits
		error("bitCount must be positive and not exceed 52")
	end
	local maxValue = 2^bitCount
	if value >= maxValue then
		error("Value out of range for " .. bitCount .. " bits. Use a higher bitCount.")
	end

	-- // Handle full bytes directly when aligned
	local bitsLeft = bitCount
	while bitsLeft >= 8 and self.bitPosition % 8 == 0 do
		local byteShift = bitsLeft - 8
		local byte = bit32.band(bit32.rshift(value, byteShift), 0xFF)

		-- // We can directly store byte to buffer to avoid bit-by-bit processing
		local bytePos = math.floor(self.bitPosition / 8) + 1
		if bytePos > #self.buffer then
			table.insert(self.buffer, byte)
		else
			self.buffer[bytePos] = byte
		end

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

	local byteIndex = bit32.rshift(self.readBitPosition, 3) + 1
	if byteIndex > #self.buffer then
		error("Attempted to read beyond buffer length")
	end
	if self.readBitPosition >= self.totalBitsWritten then
		error("Attempted to read beyond written bits")
	end
	local byte = self.buffer[byteIndex] or 0
	local bitIndex = 7 - (self.readBitPosition % 8)
	local bit = bit32.band(bit32.rshift(byte, bitIndex), 1)
	self.readBitPosition = self.readBitPosition + 1
	return bit
end

function LuaBuffer:ReadBits(bitCount)
	if not (bitCount > 0 and bitCount <= 52) then
		error("bitCount must be between 1 and 52")
	end

	-- // Flush pending bits
	if self.bitPosition % 8 ~= 0 then
		self:Flush()
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

function LuaBuffer:Flush()
	if self.bitPosition % 8 ~= 0 then -- // Store the partial byte in the buffer
		local bytePos = math.floor(self.bitPosition / 8) + 1
		if bytePos > #self.buffer then
			table.insert(self.buffer, self.currentByte)
		else
			self.buffer[bytePos] = self.currentByte
		end
		self.bytePosition = bytePos + 1
		self.currentByte = 0
	end
end

function LuaBuffer:WriteByte(byte: number)
	if byte < 0 or byte > 255 then
		error("Byte must be between 0 and 255, got " .. tostring(byte))
	end

	-- // If we're byte-aligned, write directly
	if self.bitPosition % 8 == 0 then
		local bytePos = math.floor(self.bitPosition / 8) + 1
		if bytePos > #self.buffer then
			table.insert(self.buffer, byte)
		else
			self.buffer[bytePos] = byte
		end
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
	return {table.unpack(self.buffer, 1, byteCount)}
end

---------------------------------------------
-- // Data Types
---------------------------------------------

return LuaBuffer
