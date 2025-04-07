--[[ README:

	A bit buffer module meant for data compression. Just a personal project that was fun to create;
	You might be better off using other Bit Buffer modules like BitBuf.
	
	Notes:
	
	- Bit counting is MSB to LSB
	- Recommended bitCount <32. Maximum is 52 due to rounding errors in double-percision floating points in Luau
	- Buffer growth is 1.5(bufferSize) to prevent rehashing buffer since rehashing alot causes performance issues
	- Flushing fills in the incomplete bytes with 0's; It's then stored into the buffer. (Optional optomization, you don't need to flush since it already writes only complete bytes into the buffer)
	- Reads and Writes data sequentially
	
	Example Usage:
	
local LuaBuffer = require(game.ReplicatedStorage.LuaBuffer)
local buffer = LuaBuffer.New()

buffer:WriteString("Hello World!", 16)
buffer:WriteFloat(3.2416, 20)
buffer:WriteUnsignedInteger(32002123, 32)

local base64String = buffer:EncodeBase64()

print("Base64 Encoded:", base64String)

buffer:DecodeBase64(base64String)

print(buffer:ReadString(16))
print(buffer:ReadFloat(20))
print(buffer:ReadUnsignedInteger(32))

--]]

local LuaBuffer = {}
LuaBuffer.__index = LuaBuffer

---------------------------------------------
-- // Constructor
---------------------------------------------

-- New(number) - Constructor function. Used to initalize everything
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

-- CheckSpace(number) - Checks how much space is inside of the buffer and expands it
function LuaBuffer:CheckSpace(neededCapacity: number) -- // Used to see if there's actually enough space in the buffer before writing. Creates a new table with more space and all of the contents of the old table
	if neededCapacity <= self.bufferSize then return end -- // No need to compute anything if theres already space
	-- // Geometric growth (1.5x) to balance memory usage and performance
	local newSize = math.max(math.ceil(self.bufferSize * 1.5), neededCapacity)
	local newBuffer = table.create(newSize)
	-- // Copy existing data to newBuffer
	
	table.move(self.buffer, 1, #self.buffer, 1, newBuffer)
	
	self.buffer = newBuffer
	self.bufferSize = newSize
end

---------------------------------------------
-- // Reading and Writing Bits (MSB - LSB)
---------------------------------------------

-- WriteBit() - Write a singular bit
function LuaBuffer:WriteBit(bit: number)
	if bit ~= 0 and bit ~= 1 then
		error(string.format("Bit must be 0 or 1, got %d", bit))
	end
	local shiftAmount = 7 - (self.bitPosition % 8) -- // For LSB to MSB do self.bitPosition % 8
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

-- WriteBits(number, number) - Writes multiple bits
function LuaBuffer:WriteBits(value: number, bitCount: number)
	if bitCount < 1 or bitCount > 52 then
		error(string.format("bitCount must be 1-52 (received %d) due to Lua number precision limits", bitCount))
	end
	if value < 0 then
		error(string.format("Value must be non-negative, got %d", value))
	end
	local maxValue = 2^bitCount - 1
	if value > maxValue then
		error(string.format("Value %d out of range for %d bits (max %d)", value, bitCount, maxValue))
	end

	-- // Handle full bytes directly when aligned
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

	-- // Handle remaining bits
	for i = bitCount - 1, 0, -1 do
		local bit = bit32.band(bit32.rshift(value, i), 1)
		self:WriteBit(bit)
	end
end

-- ReadBit() - Reads a singular bit
function LuaBuffer:ReadBit()
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

-- ReadBits(number) - Reads multiple bits
function LuaBuffer:ReadBits(bitCount: number)
	if bitCount < 1 or bitCount > 52 then
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
		local byteIndex = math.floor(self.readBitPosition / 8) + 1
		if byteIndex > #self.buffer then
			error("Attempted to read beyond buffer length at byteIndex " .. byteIndex)
		end
		value = bit32.lshift(value, 8) + self.buffer[byteIndex]
		self.readBitPosition = self.readBitPosition + 8
		bitsLeft = bitsLeft - 8
	end

	-- // Read remaining bits
	for _ = 1, bitsLeft do
		value = bit32.bor(bit32.lshift(value, 1), self:ReadBit())
	end

	return value
end

-- ReadByte() - Reads a full byte (8 bits) from the buffer
-- @return (number): The byte value (0-255)
function LuaBuffer:ReadByte()
	if self.readBitPosition % 8 == 0 then
		local bytePos = math.floor(self.readBitPosition / 8) + 1
		if bytePos > #self.buffer then
			error("Attempted to read beyond buffer length")
		end
		self.readBitPosition = self.readBitPosition + 8
		return self.buffer[bytePos]
	else
		return self:ReadBits(8)
	end
end

---------------------------------------------
-- // Utility Methods
---------------------------------------------

-- Reset() - Reset the bitPosition, bytePosition, currentByte, totalBitsWritten, readBitPosition, and clears the buffer
function LuaBuffer:Reset()
	self.bitPosition = 0
	self.bytePosition = 1
	self.currentByte = 0
	self.totalBitsWritten = 0
	self.readBitPosition = 0
	table.clear(self.buffer)
end

-- ResetReadPosition() - Reset readBitPosition
function LuaBuffer:ResetReadPosition()
	self.readBitPosition = 0
end

-- SeekReadPosition(number) - Allows you to move readBitPosition to a specific bit in buffer. (0-16)
function LuaBuffer:SeekReadPosition(bitPosition: number)
	if bitPosition < 0 or bitPosition > self.totalBitsWritten then
		error(string.format("Bit position %d out of range (0-%d)", 
			bitPosition, self.totalBitsWritten))
	end
	self.readBitPosition = bitPosition
end

-- Flush() - Appends incomplete bits by filling them with 0's until it makes a byte, then storing that byte
function LuaBuffer:Flush()
	if self.bitPosition % 8 ~= 0 then
		local bytePos = math.floor(self.bitPosition / 8) + 1
		self:CheckSpace(bytePos)
		self.buffer[bytePos] = self.currentByte
		self.currentByte = 0

		-- // We need to add to totalBitsWritten since we're filling in a partial byte
		local remainingBits = 8 - (self.bitPosition % 8)
		self.totalBitsWritten = math.ceil(self.bitPosition / 8) * 8
		self.currentByte = 0
		self.bitPosition = self.bitPosition + remainingBits
	end
end

-- WriteByte(number) - Writes a whole byte
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

-- GetWrittenBytes() - Returns a table containing the written bytes
function LuaBuffer:GetWrittenBytes()
	local byteCount = math.ceil(self.totalBitsWritten / 8)
	local result = table.create(byteCount)

	for i = 1, byteCount do
		result[i] = self.buffer[i] or 0
	end

	return result
end

-- GetBitPosition() - Returns the current bit position in the buffer
-- @return (number): The current bit position
function LuaBuffer:GetBitPosition()
	return self.bitPosition
end

-- GetBytePosition() - Returns the current byte position in the buffer
-- @return (number): The current byte position (derived from bitPosition)
function LuaBuffer:GetBytePosition()
	return math.floor(self.bitPosition / 8) + 1
end

---------------------------------------------
-- // Base64 Methods
---------------------------------------------

local Base64Characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

-- EncodeBase64() - Turns the buffer table into a serialized string
function LuaBuffer:EncodeBase64()
	local byteTable = self:GetWrittenBytes()
	local byteString = ""
	for i = 1, #byteTable do
		byteString = byteString .. string.char(byteTable[i])
	end

	local result = {}
	local bytePos = 1
	local len = #byteString

	while bytePos <= len do
		local b1 = byteString:byte(bytePos) or 0
		local b2 = byteString:byte(bytePos + 1) or 0
		local b3 = byteString:byte(bytePos + 2) or 0

		local num = (b1 * 65536) + (b2 * 256) + b3
		local c1 = bit32.rshift(num, 18) % 64
		local c2 = bit32.rshift(num, 12) % 64
		local c3 = bit32.rshift(num, 6) % 64
		local c4 = num % 64

		result[#result + 1] = Base64Characters:sub(c1 + 1, c1 + 1)
		result[#result + 1] = Base64Characters:sub(c2 + 1, c2 + 1)
		result[#result + 1] = Base64Characters:sub(c3 + 1, c3 + 1)
		result[#result + 1] = Base64Characters:sub(c4 + 1, c4 + 1)

		bytePos = bytePos + 3
	end

	local padding = len % 3
	if padding == 1 then
		result[#result - 1] = "="
		result[#result] = "="
	elseif padding == 2 then
		result[#result] = "="
	end

	return table.concat(result)
end

-- DecodeBase64() - Decodes the base64String into a normal buffer
-- @param base64String (string): The buffer in Base-64 string format
function LuaBuffer:DecodeBase64(base64String: string)
	local byteString = {}
	local len = string.len(base64String)

	for i = 1, len, 4 do
		local c1 = Base64Characters:find(base64String:sub(i, i)) - 1
		local c2 = Base64Characters:find(base64String:sub(i + 1, i + 1)) - 1
		local c3 = base64String:sub(i + 2, i + 2) == "=" and 0 or (Base64Characters:find(base64String:sub(i + 2, i + 2)) - 1)
		local c4 = base64String:sub(i + 3, i + 3) == "=" and 0 or (Base64Characters:find(base64String:sub(i + 3, i + 3)) - 1)

		local num = (c1 * 262144) + (c2 * 4096) + (c3 * 64) + c4
		byteString[#byteString + 1] = string.char(bit32.rshift(num, 16) % 256)
		if i + 2 <= len and base64String:sub(i + 2, i + 2) ~= "=" then
			byteString[#byteString + 1] = string.char(bit32.rshift(num, 8) % 256)
		end
		if i + 3 <= len and base64String:sub(i + 3, i + 3) ~= "=" then
			byteString[#byteString + 1] = string.char(num % 256)
		end
	end

	local decodedBytes = table.concat(byteString)
	self:Reset()
	for i = 1, #decodedBytes do
		self:WriteByte(string.byte(decodedBytes, i))
	end
end

---------------------------------------------
-- // Data Types
---------------------------------------------

-- WriteInteger() - Writes a signed integer with a specified bit count (defaults to 32-bit, range -2^(bits-1) to 2^(bits-1)-1)
-- @param value (number): The integer to write
-- @param bitCount (number?): Optional number of bits to use (defaults to 32, max 52 due to Luau precision limits)
function LuaBuffer:WriteInteger(value: number, bitCount: number?)
	local bits = bitCount or 32
	if bits < 1 or bits > 52 then
		error(string.format("bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	end
	local max = 2^(bits - 1) - 1
	local min = -2^(bits - 1)
	if value < min or value > max then
		error(string.format("Integer value %d out of range (%d to %d) for %d bits", value, min, max, bits))
	end

	self:WriteBits(value < 0 and (2^bits + value) or value, bits)
end

-- WriteUnsignedInteger() - Writes an unsigned integer with a specified bit count (defaults to 32-bit, range 0 to 2^bits-1)
-- @param value (number): The unsigned integer to write
-- @param bitCount (number?): Optional number of bits to use (defaults to 32, max 52 due to Luau precision limits)
function LuaBuffer:WriteUnsignedInteger(value: number, bitCount: number?)
	local bits = bitCount or 32
	if bits < 1 or bits > 52 then
		error(string.format("bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	end
	local max = 2^bits - 1
	if value < 0 or value > max then
		error(string.format("Unsigned integer value %d out of range (0 to %d) for %d bits", value, max, bits))
	end
	self:WriteBits(value, bits)
end

-- WriteBool() - Writes a boolean as 1 bit (true = 1, false = 0)
-- @param value (boolean): The boolean to write
function LuaBuffer:WriteBool(value: boolean)
	self:WriteBit(value and 1 or 0)
end

-- WriteString() - Writes a string with a length prefix (defaults to 16-bit length, each character as 8-bit bytes)
-- @param value (string): The string to write
-- @param lengthBits (number?): Optional number of bits for the length prefix (defaults to 16, max 52)
function LuaBuffer:WriteString(value: string, lengthBits: number?)
	local len = #value
	local bits = lengthBits or 16
	if bits < 1 or bits > 52 then
		error(string.format("lengthBits must be 1-52 (received %d) due to Lua number precision limits", bits))
	end
	local maxLength = 2^bits - 1
	if len > maxLength then
		error(string.format("String length %d exceeds maximum (%d) for %d-bit length prefix", len, maxLength, bits))
	end
	self:WriteBits(len, bits)
	for i = 1, len do
		self:WriteBits(string.byte(value, i), 8)
	end
end

-- WriteFloat() - Writes a 32-bit (single precision) floating point number. (Sign: 1 Bit, biasedExponent: 8 Bits, mantissaBits: 23)
-- @param value (number): The float to write
function LuaBuffer:WriteFloat(value: number)
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
		
		-- // There's an edge case when writing the mantissa_bits where it can round to exactly 2^23
		local log_val = math.log(abs_value, 2)
		
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

-- ReadInteger() - Reads a signed integer with a specified bit count (defaults to 32-bit)
-- @param bitCount (number?): Optional number of bits to read (defaults to 32, max 52)
-- @return (number): The signed integer value
function LuaBuffer:ReadInteger(bitCount: number?): number
	local bits = bitCount or 32
	if bits < 1 or bits > 52 then
		error(string.format("bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	end
	local value = self:ReadBits(bits)
	local threshold = 2^(bits - 1)
	-- // Convert to signed value if above threshold
	return value >= threshold and (value - 2^bits) or value
end

-- ReadUnsignedInteger() - Reads an unsigned integer with a specified bit count (defaults to 32-bit)
-- @param bitCount (number?): Optional number of bits to read (defaults to 32, max 52)
-- @return (number): The unsigned integer value
function LuaBuffer:ReadUnsignedInteger(bitCount: number?): number
	local bits = bitCount or 32
	if bits < 1 or bits > 52 then
		error(string.format("bitCount must be 1-52 (received %d) due to Lua number precision limits", bits))
	end
	return self:ReadBits(bits)
end

-- ReadBool() - Reads a boolean as 1 bit
-- @return (boolean): The boolean value (true if 1, false if 0)
function LuaBuffer:ReadBool(): boolean
	return self:ReadBit() == 1
end

-- ReadString() - Reads a string with a length prefix (defaults to 16-bit length, each character as 8-bit bytes)
-- @param lengthBits (number?): Optional number of bits for the length prefix (defaults to 16, max 52)
-- @return (string): The read string
function LuaBuffer:ReadString(lengthBits: number?): string
	local bits = lengthBits or 16
	if bits < 1 or bits > 52 then
		error(string.format("lengthBits must be 1-52 (received %d) due to Lua number precision limits", bits))
	end
	local len = self:ReadBits(bits)
	local bytes = {}
	for i = 1, len do
		bytes[i] = self:ReadByte()
	end
	return string.char(table.unpack(bytes))
end

-- ReadFloat() - Reads a 32-bit (single precision) floating point number. (Sign: 1 Bit, biasedExponent: 8 Bits, mantissaBits: 23)
-- @param value (number): The float to read
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
