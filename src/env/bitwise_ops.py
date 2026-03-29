# Core math logic for AI Hash Architect
def rotr(x, n):
    """Rotate right (32-bit integer)."""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def rotl(x, n):
    """Rotate left (32-bit integer)."""
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

def ch(x, y, z):
    """Choose function."""
    return (x & y) ^ (~x & z)

def maj(x, y, z):
    """Majority function."""
    return (x & y) ^ (x & z) ^ (y & z)

def sigma0_256(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)

def sigma1_256(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

def lower_sigma0_256(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)

def lower_sigma1_256(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)

def add_mod32(a, b):
    return (a + b) & 0xFFFFFFFF
