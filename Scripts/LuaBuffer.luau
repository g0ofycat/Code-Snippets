--!strict

--[[ README:

	A bit buffer module meant for data compression. Just a personal project that was fun to create
	
	NOTES:
	
	- Bit counting is MSB to LSB
	- Recommended bitCount <32. Maximum is 52 due to rounding errors in double-percision floating points in Luau
	- Buffer growth is 1.5(bufferSize) to prevent rehashing buffer since rehashing alot causes performance issues
	- Flushing fills in the incomplete bytes with 0's; It's then stored into the buffer.
	- Reads and Writes data sequentially
]]

--===========================
-- // TYPES
--===========================

export type LuaBufferImpl = {
	__index: LuaBufferImpl,
	New: (initialSize: number?) -> LuaBuffer,
	CheckSpace: (self: LuaBuffer, neededCapacity: number) -> (),
	WriteBit: (self: LuaBuffer, bit: number) -> (),
	WriteBits: (self: LuaBuffer, value: number, bitCount: number) -> (),
	ReadBit: (self: LuaBuffer) -> number,
	ReadBits: (self: LuaBuffer, bitCount: number) -> number,
	ReadByte: (self: LuaBuffer) -> number,
	Reset: (self: LuaBuffer) -> (),
	ResetReadPosition: (self: LuaBuffer) -> (),
	SeekReadPosition: (self: LuaBuffer, bitPosition: number) -> (),
	Flush: (self: LuaBuffer) -> (),
	WriteByte: (self: LuaBuffer, byte: number) -> (),
	GetWrittenBytes: (self: LuaBuffer) -> {number},
	GetBitPosition: (self: LuaBuffer) -> number,
	GetBytePosition: (self: LuaBuffer) -> number,
	WriteInteger: (self: LuaBuffer, value: number, bitCount: number?) -> (),
	WriteUnsignedInteger: (self: LuaBuffer, value: number, bitCount: number?) -> (),
	WriteBool: (self: LuaBuffer, value: boolean) -> (),
	WriteString: (self: LuaBuffer, value: string, lengthBits: number?) -> (),
	WriteFloat: (self: LuaBuffer, value: number) -> (),
	ReadInteger: (self: LuaBuffer, bitCount: number?) -> number,
	ReadUnsignedInteger: (self: LuaBuffer, bitCount: number?) -> number,
	ReadBool: (self: LuaBuffer) -> boolean,
	ReadString: (self: LuaBuffer, lengthBits: number?) -> string,
	ReadFloat: (self: LuaBuffer) -> number,
}

export type LuaBuffer = typeof(setmetatable({} :: {
	buffer: {number},
	bufferSize: number,
	bitPosition: number,
	bytePosition: number,
	currentByte: number,
	totalBitsWritten: number,
	readBitPosition: number,
}, {} :: LuaBufferImpl))

local LuaBuffer = {} :: LuaBufferImpl
LuaBuffer.__index = LuaBuffer

--===========================
-- // CONSTRUCTOR
--===========================

-- New(): Constructor function
-- @param initialSize: Initial size of the buffer in bytes (default: 32)
-- @return LuaBuffer: self
function LuaBuffer.New(initialSize: number?): LuaBuffer
	initialSize = initialSize or 32
	
	assert(initialSize > 0, "New(): initialSize must be positive")

	return setmetatable({
		buffer = table.create(initialSize :: number), -- // The reason we are using table.create() instead of {} is to avoid rehashing (The resizing of tables). When inserting a value into a table, it may expand causing performance issues
		bufferSize = initialSize, -- // Tracks buffer capacity, used in CheckSpace to see if we can write in buffer
		bitPosition = 0, -- // Tracks the current bit position within the buffer, when it reaches a multiple of 8, the byte is stored
		bytePosition = 1, -- // The current position of the byte, allows us to see if the next byte should be inserted into the buffer by checking if it's too large to hold the new byte. If not then just overwrite the existing byte at bytePosition
		currentByte = 0, -- // The actual byte we're on in WriteBit. Used to store and overwrite specific bytes in the buffer table
		totalBitsWritten = 0, -- // Tracks total bits written to limit reads to valid data. (Which are fully written bytes)
		readBitPosition = 0 -- // Tracks read position separately from write position
	}, LuaBuffer) :: LuaBuffer
end

--===========================
-- // HELPER FUNCTIONS
--===========================

-- CheckSpace(): Checks how much space is inside of the buffer and expands it
-- @param neededCapacity: The capacity needed in bytes
function LuaBuffer:CheckSpace(neededCapacity: number): ()
	if neededCapacity <= self.bufferSize then return end
	local newSize = math.max(math.ceil(self.bufferSize * 1.5), neededCapacity)
	local newBuffer = table.create(newSize)

	table.move(self.buffer, 1, #self.buffer, 1, newBuffer)

	self.buffer = newBuffer
	self.bufferSize = newSize
end

--===========================
-- // READING & WRITING (MSB - LSB)
--===========================

-- WriteBit(): Write a singular bit
-- @param bit: The bit value (0 or 1)
function LuaBuffer:WriteBit(bit: number): ()
	assert(bit == 0 or bit == 1, string.format("WriteBit(): Bit must be 0 or 1, got %d", bit))
	
	local shiftAmount = 7 - (self.bitPosition % 8) -- // LSB - MSB: self.bitPosition % 8
	local shiftedBit = bit32.lshift(bit, shiftAmount)
	self.currentByte = bit32.bor(self.currentByte, shiftedBit)
	self.bitPosition = self.bitPosition + 1
	self.totalBitsWritten = self.totalBitsWritten + 1
	
	if self.bitPosition % 8 == 0 then
		local bytePos = math.floor(self.bitPosition / 8)
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = self.currentByte
		self.bytePosition = bytePos + 1
		self.currentByte = 0
	end
end

-- WriteBits(): Writes multiple bits
-- @param value: The value to write
-- @param bitCount: Number of bits to write (1-52)
function LuaBuffer:WriteBits(value: number, bitCount: number): ()
	assert(bitCount >= 1 and bitCount <= 52, string.format("WriteBits(): bitCount must be 1-52 (received %d) due to Lua number precision limits", bitCount))
	assert(value >= 0, string.format("WriteBits(): Value must be non-negative, got %d", value))
	local maxValue = 2^bitCount - 1
	assert(value <= maxValue, string.format("WriteBits(): Value %d out of range for %d bits (max %d)", value, bitCount, maxValue))

	while bitCount >= 8 and self.bitPosition % 8 == 0 do
		local byteShift = bitCount - 8
		local byte = bit32.band(bit32.rshift(value, byteShift), 255)
		local bytePos = math.floor(self.bitPosition / 8) + 1
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = byte
		self.bitPosition = self.bitPosition + 8
		self.totalBitsWritten = self.totalBitsWritten + 8
		self.bytePosition = bytePos + 1
		bitCount = bitCount - 8
		value = bit32.band(value, (2^bitCount - 1))
	end

	for i = bitCount - 1, 0, -1 do
		local bit = bit32.band(bit32.rshift(value, i), 1)
		self:WriteBit(bit)
	end
end

-- ReadBit(): Reads a singular bit
-- @return number: The bit value (0 or 1)
function LuaBuffer:ReadBit(): number
	local byteIndex = bit32.rshift(self.readBitPosition, 3) + 1
	
	assert(self.readBitPosition < self.totalBitsWritten, "ReadBit(): Attempted to read beyond written bits")
	assert(byteIndex <= #self.buffer and self.buffer[byteIndex] ~= nil, "ReadBit(): Attempted to read beyond buffer length or from nil byte at byteIndex " .. byteIndex)

	local byte = self.buffer[byteIndex]
	local bitIndex = 7 - (self.readBitPosition % 8)
	local bit = bit32.band(bit32.rshift(byte, bitIndex), 1)
	
	self.readBitPosition = self.readBitPosition + 1
	
	return bit
end

-- ReadBits(): Reads multiple bits
-- @param bitCount: Number of bits to read (1-52)
-- @return number: The value read
function LuaBuffer:ReadBits(bitCount: number): number
	assert(self.bitPosition % 8 == 0, "ReadBits(): Buffer has partial bytes; call Flush() before reading")
	assert(bitCount >= 1 and bitCount <= 52, string.format("ReadBits(): bitCount must be 1-52 (received %d) due to Lua number precision limits", bitCount))
	
	local availableBits = self.totalBitsWritten - self.readBitPosition
	
	assert(bitCount <= availableBits, string.format("ReadBits(): Requested %d bits, only %d available", bitCount, availableBits))

	local value = 0
	local bitsLeft = bitCount

	while bitsLeft >= 8 and self.readBitPosition % 8 == 0 do
		local byteIndex = math.floor(self.readBitPosition / 8) + 1
		
		assert(byteIndex <= #self.buffer, "ReadBits(): Attempted to read beyond buffer length at byteIndex " .. byteIndex)
		
		value = bit32.lshift(value, 8) + self.buffer[byteIndex]
		self.readBitPosition = self.readBitPosition + 8
		bitsLeft = bitsLeft - 8
	end

	for _ = 1, bitsLeft do
		value = bit32.bor(bit32.lshift(value, 1), self:ReadBit())
	end

	return value
end

-- ReadByte(): Reads a full byte (8 bits) from the buffer
-- @return number: The byte value (0-255)
function LuaBuffer:ReadByte(): number
	if self.readBitPosition % 8 == 0 then
		local bytePos = math.floor(self.readBitPosition / 8) + 1
		
		assert(bytePos <= #self.buffer, "ReadByte(): Attempted to read beyond buffer length")
		
		self.readBitPosition = self.readBitPosition + 8
		return self.buffer[bytePos]
	else
		return self:ReadBits(8)
	end
end

--===========================
-- // UTILITY METHODS
--===========================

-- Reset(): Reset the bitPosition, bytePosition, currentByte, totalBitsWritten, readBitPosition, and clears the buffer
function LuaBuffer:Reset(): ()
	self.bitPosition = 0
	self.bytePosition = 1
	self.currentByte = 0
	self.totalBitsWritten = 0
	self.readBitPosition = 0
	
	table.clear(self.buffer)
end

-- ResetReadPosition(): Resets readBitPosition
function LuaBuffer:ResetReadPosition(): ()
	self.readBitPosition = 0
end

-- SeekReadPosition(): Allows you to move readBitPosition to a specific bit in buffer
-- @param bitPosition: The bit position to seek to (0 to totalBitsWritten)
function LuaBuffer:SeekReadPosition(bitPosition: number): ()
	assert(bitPosition >= 0 and bitPosition <= self.totalBitsWritten, string.format("SeekReadPosition(): Bit position %d out of range (0-%d)", bitPosition, self.totalBitsWritten))
	
	self.readBitPosition = bitPosition
end

-- Flush(): Appends incomplete bits by filling them with 0's until it makes a byte, then storing that byte. Only needs to be called before reading if there's a partial byte at the end
function LuaBuffer:Flush(): ()
	if self.bitPosition % 8 ~= 0 then
		local bytePos = math.floor(self.bitPosition / 8) + 1
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = self.currentByte
		self.currentByte = 0

		local remainingBits = 8 - (self.bitPosition % 8)
		self.totalBitsWritten = math.ceil(self.bitPosition / 8) * 8
		self.currentByte = 0
		self.bitPosition = self.bitPosition + remainingBits
	end
end

-- WriteByte(): Writes a whole byte
-- @param byte: The byte value to write (0-255)
function LuaBuffer:WriteByte(byte: number): ()
	assert(byte >= 0 and byte <= 255, string.format("WriteByte(): Byte must be between 0 and 255, got %d", byte))
	
	if self.bitPosition % 8 == 0 then
		local bytePos = math.floor(self.bitPosition / 8) + 1
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = byte
		self.bitPosition = self.bitPosition + 8
		self.totalBitsWritten = self.totalBitsWritten + 8
		self.bytePosition = bytePos + 1
	else
		for i = 7, 0, -1 do
			self:WriteBit(bit32.band(bit32.rshift(byte, i), 1))
		end
	end
end

-- GetWrittenBytes(): Returns a table containing the written bytes
-- @return table: A table of bytes
function LuaBuffer:GetWrittenBytes(): { number }
	self:Flush()
	local byteCount = math.ceil(self.totalBitsWritten / 8)
	return table.move(self.buffer, 1, byteCount, 1, table.create(byteCount, 0))
end

-- GetBitPosition(): Returns the current bit position in the buffer
-- @return number: The current bit position
function LuaBuffer:GetBitPosition(): number
	return self.bitPosition
end

-- GetBytePosition(): Returns the current byte position in the buffer
-- @return number: The current byte position (derived from bitPosition)
function LuaBuffer:GetBytePosition(): number
	return math.floor(self.bitPosition :: number / 8) + 1
end

--===========================
-- // DATA TYPES
--===========================

-- WriteInteger(): Writes a signed integer with a specified bit count (defaults to 32-bit, range -2^(bits-1) to 2^(bits-1)-1)
-- @param value: The integer to write
-- @param bitCount: Optional number of bits to use (defaults to 32, max 52 due to Luau precision limits)
function LuaBuffer:WriteInteger(value: number, bitCount: number?): ()
	local bits = bitCount or 32
	assert(bits >= 1 and bits <= 52, string.format("WriteInteger(): bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	
	local max = 2^(bits - 1) - 1
	local min = -2^(bits - 1)
	
	assert(value >= min and value <= max, string.format("WriteInteger(): Integer value %d out of range (%d to %d) for %d bits", value, min, max, bits))

	self:WriteBits(value < 0 and (2^bits + value) or value, bits)
end

-- WriteUnsignedInteger(): Writes an unsigned integer with a specified bit count (defaults to 32-bit, range 0 to 2^bits-1)
-- @param value: The unsigned integer to write
-- @param bitCount: Optional number of bits to use (defaults to 32, max 52 due to Luau precision limits)
function LuaBuffer:WriteUnsignedInteger(value: number, bitCount: number?): ()
	local bits = bitCount or 32
	
	assert(bits >= 1 and bits <= 52, string.format("WriteUnsignedInteger(): bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	
	local max = 2^bits - 1
	
	assert(value >= 0 and value <= max, string.format("WriteUnsignedInteger(): Unsigned integer value %d out of range (0 to %d) for %d bits", value, max, bits))
	
	self:WriteBits(value, bits)
end

-- WriteBool(): Writes a boolean as 1 bit (true = 1, false = 0)
-- @param value: The boolean to write
function LuaBuffer:WriteBool(value: boolean): ()
	self:WriteBit(value and 1 or 0)
end

-- WriteString(): Writes a string with a length prefix (defaults to 16-bit length, each character as 8-bit bytes)
-- @param value: The string to write
-- @param lengthBits?: Optional number of bits for the length prefix (defaults to 16, max 52)
function LuaBuffer:WriteString(value: string, lengthBits: number?): ()
	local len = #value
	local bits = lengthBits or 16
	
	assert(bits >= 1 and bits <= 52, string.format("WriteString(): lengthBits must be 1-52 (received %d) due to Lua number precision limits", bits))
	
	local maxLength = 2^bits - 1
	
	assert(len <= maxLength, string.format("WriteString(): String length %d exceeds maximum (%d) for %d-bit length prefix", len, maxLength, bits))
	
	self:WriteBits(len, bits)
	
	for i = 1, len do
		self:WriteBits(string.byte(value, i), 8)
	end
end

-- WriteFloat(): Writes a 32-bit (single precision) floating point number. (Sign: 1 Bit, biasedExponent: 8 Bits, mantissaBits: 23)
-- @param value: The float to write
function LuaBuffer:WriteFloat(value: number): ()
	local sign = value < 0 and 1 or 0
	local abs_value = math.abs(value)
	local smallest_normalized = math.pow(2, -126)

	if abs_value == 0 then
		self:WriteBit(sign)
		self:WriteBits(0, 8)
		self:WriteBits(0, 23)
		return
	elseif abs_value == math.huge then
		self:WriteBit(sign)
		self:WriteBits(255, 8)
		self:WriteBits(0, 23)
		return
	elseif value ~= value then
		self:WriteBit(sign)
		self:WriteBits(255, 8)
		self:WriteBits(1, 23)
		return
	end

	if abs_value >= smallest_normalized then
		local log_val = math.log(abs_value, 2)
		local exponent = math.floor(log_val)
		local fraction = (abs_value / math.pow(2, exponent)) - 1
		local biased_exponent = exponent + 127

		if biased_exponent > 254 then
			self:WriteBit(sign)
			self:WriteBits(255, 8)
			self:WriteBits(0, 23)
			return
		end

		-- // edge case when writing the mantissa_bits where it can round to exactly 2^23
		
		local mantissa_bits = math.round(fraction * math.pow(2, 23))
		self:WriteBit(sign)
		self:WriteBits(biased_exponent, 8)
		self:WriteBits(mantissa_bits, 23)
	else
		local fraction = abs_value / math.pow(2, -126)
		local mantissa_bits = math.round(fraction * math.pow(2, 23))
		if mantissa_bits >= math.pow(2, 23) then
			local exponent = -126
			local biased_exponent = exponent + 127
			local fraction_part = (abs_value / math.pow(2, exponent)) - 1
			local mantissa_bits_normal = math.round(fraction_part * math.pow(2, 23))
			self:WriteBit(sign)
			self:WriteBits(biased_exponent, 8)
			self:WriteBits(mantissa_bits_normal, 23)
		else
			self:WriteBit(sign)
			self:WriteBits(0, 8)
			self:WriteBits(mantissa_bits, 23)
		end
	end
end

-- ReadInteger(): Reads a signed integer with a specified bit count (defaults to 32-bit)
-- @param bitCount: Optional number of bits to read (defaults to 32, max 52)
-- @return number: The signed integer value
function LuaBuffer:ReadInteger(bitCount: number?): number
	local bits = bitCount or 32
	
	assert(bits >= 1 and bits <= 52, string.format("ReadInteger(): bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	
	local value = self:ReadBits(bits)
	local threshold = 2^(bits - 1)
	
	return value >= threshold and (value - 2^bits) or value -- // convert to signed value if above threshold
end

-- ReadUnsignedInteger(): Reads an unsigned integer with a specified bit count (defaults to 32-bit)
-- @param bitCount: Optional number of bits to read (defaults to 32, max 52)
-- @return number: The unsigned integer value
function LuaBuffer:ReadUnsignedInteger(bitCount: number?): number
	local bits = bitCount or 32
	
	assert(bits >= 1 and bits <= 52, string.format("ReadUnsignedInteger(): bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	
	return self:ReadBits(bits)
end

-- ReadBool(): Reads a boolean as 1 bit
-- @return boolean: The boolean value (true if 1, false if 0)
function LuaBuffer:ReadBool(): boolean
	return self:ReadBit() == 1
end

-- ReadString(): Reads a string with a length prefix (defaults to 16-bit length, each character as 8-bit bytes)
-- @param lengthBits: Optional number of bits for the length prefix (defaults to 16, max 52)
-- @return string: The read string
function LuaBuffer:ReadString(lengthBits: number?): string
	local bits = lengthBits or 16
	
	assert(bits >= 1 and bits <= 52, string.format("ReadString(): lengthBits must be 1-52 (received %d) due to Lua number precision limits", bits))
	
	local len = self:ReadBits(bits)
	
	local bytes = {}
	
	for i = 1, len do
		bytes[i] = self:ReadByte()
	end
	
	return string.char(table.unpack(bytes))
end

-- ReadFloat(): Reads a 32-bit (single precision) floating point number. (Sign: 1 Bit, biasedExponent: 8 Bits, mantissaBits: 23)
-- @return number: The float value
function LuaBuffer:ReadFloat(): number
	local sign = self:ReadBit()
	local exponent = self:ReadByte()
	local mantissa = self:ReadBits(23)

	if exponent == 0 then
		if mantissa == 0 then
			return sign == 1 and -0 or 0
		else
			return (sign == 1 and -1 or 1) * (mantissa / (2^23)) * (2^(-126))
		end
	elseif exponent == 255 then
		if mantissa == 0 then
			return sign == 1 and -math.huge or math.huge
		else
			return 0/0
		end
	end

	local value = (1 + mantissa / (2^23)) * (2^(exponent - 127))

	return sign == 1 and -value or value
end

return LuaBuffer
