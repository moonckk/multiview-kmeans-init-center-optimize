# coding=UTF-8

class code:
    def _init_(self):
        self = self

    def getBin(self, x):
        return x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]

    def floatToBinary64(self, value):

        if value == 0:
            return '0000000000000000000000000000000000000000000000000000000000000000'
        else:
            val = struct.unpack('Q', struct.pack('d', value))[0]
            return self.getBin(val).rjust(64, '0')

    def binaryToFloat(self, value):
        sign = value.rjust(64, '0')[0]
        value = value.rjust(64, '0')[1:]
        hx = hex(int(value, 2))
        result = struct.unpack("d", struct.pack("q", int(hx, 16)))[0]
        return sign == '1' and -1 * result or result


import struct

getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]


def floatToBinary64(value):
    val = struct.unpack('Q', struct.pack('d', value))[0]
    return getBin(val)


def binaryToFloat(value):
    sign = value.rjust(64, '0')[0]

    value = value.rjust(64, '0')[1:]
    hx = hex(int(value, 2))
    result = struct.unpack("d", struct.pack("q", int(hx, 16)))[0]
    return sign == '1' and -1 * result or result


if __name__ == "__main__":
    acod = floatToBinary64(-1.000002)
    bcod = floatToBinary64(-21.2)
    print(len(acod.rjust(64, '0')), len(bcod), acod.rjust(64, '0'), bcod)
    print(binaryToFloat(acod))
    print(code().binaryToFloat(acod))
