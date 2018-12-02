import io
import os
import string
import time
from collections import deque


def rotate_left(value, rotate_num_bits, max_bit_length):
    """
    example code
    rotate left
    0b11011 --> 0b10111
    """
    x = rotate_num_bits % max_bit_length
    y = max_bit_length - x
    z = 2 ** max_bit_length - 1
    return ((value << x) & z) | ((value & z) >> y)


def rotate_right(value, rotate_num_bits, max_bit_length):
    """
    example code
    rotate right
    0b11011 --> 0b11101
    """
    x = rotate_num_bits % max_bit_length
    y = max_bit_length - x
    z = 2 ** max_bit_length - 1
    return ((value << y) & z) | ((value & z) >> x)


def memo(f):
    """
    memoization decorator using dict subclass
    only for single-argument functions
    """

    class Memo(dict):
        def __missing__(self, key):
            result = self[key] = f(key)
            return result

    return Memo().__getitem__


@memo
def char_hash(char):
    """
    hash a unicode char

    could also just return ord(char) but ideally the hash should be uniformly-distributed over the output range
    which is means it should be a random 32-bit integer

    come to think of it there's probably an ideal distribution for these hashes which will be language dependent
    but meh

    :return: int_32
    """
    return hash(char)


class BuzHash(object):
    """
    rolling hash using a cyclic polynomial
    see: https://en.wikipedia.org/wiki/Rolling_hash#Cyclic_polynomial

    there is no benefit to having max_bit_length > hash_bit_len unless rotate_num_bits * item_len > hash_bit_len
    where hash_bit_len is 32 for python hashing

    this is also a bad hash for anything with a very limited input character sequence
    like DNA matching or binary
    and especially bad when used on the thue-morse sequence 01101001100101101001011001101001....
    see: http://www.ioinformatics.org/oi/pdf/INFOL119.pdf
    """

    def __init__(self, item_len, init_chars='', rotate_num_bits=19, max_bit_length=32, left_rotate=False):
        """
        ideal case:
        (rotate_num_bits * 1.618 == max_bit_length) & (gcd(rotate_num_bits, max_bit_length) == 1)
        
        TODO: to approximate phi more closely, go to this o'clock at each step: [int((.5*(5**.5+1)*i*32)%31) for i in range(32*32)]
        the reason to approximate phi is that phi is the hardest irrational number to approximate with a fraction
        it might be close enough to fairly distributed around a circle, while never really repeating
        
        but it's not that fairly distributed so if there are problems then maybe try this: [round((.5*(5**.5+1)*i*32)%32)%32 for i in range(34*75)]
        this kind of cycles every 34 steps instead of 32
        the smaller cycle kind of cycles every 34*75 steps
        """
        # pre-compute some stuff
        self.x1 = rotate_num_bits % max_bit_length
        self.y1 = max_bit_length - self.x1
        self.x2 = (item_len * rotate_num_bits) % max_bit_length
        self.y2 = max_bit_length - self.x2
        self.z = 2 ** max_bit_length - 1

        # to rotate left instead of right, swap the x and y values
        if left_rotate:
            self.x1, self.y1, self.x2, self.y2 = self.y1, self.x1, self.y2, self.x2

        # init storage
        self.hash_val = 0
        self.circular_buffer = deque([0] * item_len, maxlen=item_len)
        self.char_buffer = deque([None] * item_len, maxlen=item_len)

        # init hash
        self.transduce(init_chars)

    def step(self, char, verbose=False):
        """
        step forward hash by one char
        takes about 2e-6 seconds per step, equivalent to about 2.2 secs/MiB
        :param char:
        :param verbose:
        :return:
        """
        self.hash_val = ((self.hash_val << self.y1) & self.z) | ((self.hash_val & self.z) >> self.x1)
        # delete old item from window (must precede saving the next item, because the buffer has a limited length)
        self.hash_val ^= self.circular_buffer.popleft()
        # hash new item
        char_val = char_hash(char) & self.z  # necessary as long as max_bit_length != 32
        # add new item to window
        self.hash_val ^= char_val
        # save pre-rotated hash to the circular buffer so it can be removed later
        self.circular_buffer.append(((char_val << self.y2) & self.z) | ((char_val & self.z) >> self.x2))
        self.char_buffer.append(char)
        # maybe print
        if verbose:
            print(char, self.hash_val)
        # current hash value
        return self.hash_val

    def transduce(self, seq, verbose=False):
        return [self.step(item, verbose=verbose) for item in seq]


def de_bruijn(alphabet, sub_seq_len, unwrap=True):
    """
    generate a sequence containing all possible sub-sequences of some length given an alphabet
    eg: de_bruijn('abc', 3, True)  --> 'aaabaacabbabcacbaccbbbcbcccaa'
    eg: de_bruijn('abc', 3, False) --> 'aaabaacabbabcacbaccbbbcbccc'

    :param alphabet: string of chars, or iterable of objects
    :param sub_seq_len:
    """
    alphabet = sorted(set(alphabet))
    a = [0] * len(alphabet) * sub_seq_len
    buffer = []

    def db(t, p):
        if t > sub_seq_len:
            if sub_seq_len % p == 0:
                for char_index in a[1:p + 1]:
                    yield alphabet[char_index]
        else:
            a[t] = a[t - p]

            for char in db(t + 1, p):
                yield char

            for j in range(a[t - p] + 1, len(alphabet)):
                a[t] = j
                for char in db(t + 1, t):
                    yield char

    gen = db(1, 1)
    if unwrap:
        for _ in range(sub_seq_len - 1):
            out = gen.__next__()
            buffer.append(out)
            yield out

    for out in gen:
        yield out

    if unwrap:
        for out in buffer:
            yield out


if __name__ == '__main__':

    assert rotate_left(0b11011, 1, 5) == 0b10111 == rotate_right(0b11011, -1, 5)
    assert rotate_left(0b11001, 1, 5) == 0b10011 == rotate_right(0b11001, -1, 5)
    assert rotate_left(0b11010, 1, 5) == 0b10101 == rotate_right(0b11010, -1, 5)
    assert rotate_right(0b11011, 1, 5) == 0b11101 == rotate_left(0b11011, -1, 5)
    assert rotate_right(0b10011, 1, 5) == 0b11001 == rotate_left(0b10011, -1, 5)
    assert rotate_right(0b11001, 1, 5) == 0b11100 == rotate_left(0b11001, -1, 5)

    char_hashes = {u'\t':   1152003464,
                   u'\n':   1280003851,
                   u'\x0b': 1408004234,
                   u'\x0c': 1536004621,
                   u'\r':   1664005004,
                   u' ':    -198954975,
                   u'!':    -70954592,
                   u'"':    57045795,
                   u'#':    185046178,
                   u'$':    313046565,
                   u'%':    441046948,
                   u'&':    569047335,
                   u"'":    697047718,
                   u'(':    825048105,
                   u')':    953048488,
                   u'*':    1081048875,
                   u'+':    1209049258,
                   u',':    1337049645,
                   u'-':    1465050028,
                   u'.':    1593050415,
                   u'/':    1721050798,
                   u'0':    1849051185,
                   u'1':    1977051568,
                   u'2':    2105051955,
                   u'3':    -2061914958,
                   u'4':    -1933914571,
                   u'5':    -1805914188,
                   u'6':    -1677913801,
                   u'7':    -1549913418,
                   u'8':    -1421913031,
                   u'9':    -1293912648,
                   u':':    -1165912261,
                   u';':    -1037911878,
                   u'<':    -909911491,
                   u'=':    -781911108,
                   u'>':    -653910721,
                   u'?':    -525910338,
                   u'@':    -397909951,
                   u'A':    -269909568,
                   u'B':    -141909181,
                   u'C':    -13908798,
                   u'D':    114091589,
                   u'E':    242091972,
                   u'F':    370092359,
                   u'G':    498092742,
                   u'H':    626093129,
                   u'I':    754093512,
                   u'J':    882093899,
                   u'K':    1010094282,
                   u'L':    1138094669,
                   u'M':    1266095052,
                   u'N':    1394095439,
                   u'O':    1522095822,
                   u'P':    1650096209,
                   u'Q':    1778096592,
                   u'R':    1906096979,
                   u'S':    2034097362,
                   u'T':    -2132869547,
                   u'U':    -2004869164,
                   u'V':    -1876868777,
                   u'W':    -1748868394,
                   u'X':    -1620868007,
                   u'Y':    -1492867624,
                   u'Z':    -1364867237,
                   u'[':    -1236866854,
                   u'\\':   -1108866467,
                   u']':    -980866084,
                   u'^':    -852865697,
                   u'_':    -724865314,
                   u'`':    -596864927,
                   u'a':    -468864544,
                   u'b':    -340864157,
                   u'c':    -212863774,
                   u'd':    -84863387,
                   u'e':    43136996,
                   u'f':    171137383,
                   u'g':    299137766,
                   u'h':    427138153,
                   u'i':    555138536,
                   u'j':    683138923,
                   u'k':    811139306,
                   u'l':    939139693,
                   u'm':    1067140076,
                   u'n':    1195140463,
                   u'o':    1323140846,
                   u'p':    1451141233,
                   u'q':    1579141616,
                   u'r':    1707142003,
                   u's':    1835142386,
                   u't':    1963142773,
                   u'u':    2091143156,
                   u'v':    -2075823753,
                   u'w':    -1947823370,
                   u'x':    -1819822983,
                   u'y':    -1691822600,
                   u'z':    -1563822213,
                   u'{':    -1435821830,
                   u'|':    -1307821443,
                   u'}':    -1179821060,
                   u'~':    -1051820673}

    h_set = set()

    # for char in char_hashes:
    #     assert char_hash(char) == char_hashes[char]

    seq_len = 3
    bh = BuzHash(seq_len, 'asdfgh', max_bit_length=32)
    h = bh.hash_val
    c = ''.join(bh.char_buffer)

    temp = bh.transduce('asdfgh')
    assert bh.transduce('asdfghasdfgh') == temp + temp

    if seq_len > 3:
        print('WARNING: this is gonna take really long')
    t = time.time()
    i = 0
    j = 0
    for char in de_bruijn(string.printable, seq_len):
        i += 1
        hh = bh.step(char)
        h_set.add(hh)
        if hh == h:
            j += 1
            if ''.join(bh.char_buffer) != c:
                print('hash collision: "%s", "%s"' % (c, ''.join(bh.char_buffer)))

    print(time.time() - t, 'seconds')
    print(i, 'items tested')
    print(j, 'items matched')

    t = time.time()
    bh = BuzHash(6, max_bit_length=32)
    with io.open('kjv.txt', mode='r', encoding='utf8') as f:
        for chunk in iter(lambda: f.read(65335), ''):
            for char in chunk:
                h = bh.step(char)
                if h in h_set:
                    # print(h)
                    pass
    print(time.time() - t)
    print(os.path.getsize('kjv.txt'))

