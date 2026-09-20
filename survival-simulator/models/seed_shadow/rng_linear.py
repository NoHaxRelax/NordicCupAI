"""MT19937 state inference from truncated outputs at KNOWN stream positions.

This is an experimental inference primitive, not an in-game seed finder. The
game's missing draws and unordered/rounded geometry must be resolved separately.
It uses GF(2) equations instead of enumerating seeds. A word is 32 symbolic bits;
each symbolic bit is a Python integer bitset over 624*32 unknown state bits.
The transition/tempering constants are those of CPython's _randommodule.c.
"""
import math
import random

NVAR=624*32


def xor(a,b):
    return [x^y for x,y in zip(a,b)]


def right(a,n):
    return a[n:]+[0]*n


def left_mask(a,n,mask):
    return [a[i-n] if i>=n and (mask>>i)&1 else 0 for i in range(32)]


class MTLinear:
    def __init__(self):
        self.initial=[[1<<(word*32+bit) for bit in range(32)] for word in range(624)]
        self.state=[w[:] for w in self.initial]
        self.index=0
        self.words=0
        self.basis={}

    def equation(self,row,rhs):
        while row:
            pivot=row.bit_length()-1
            old=self.basis.get(pivot)
            if old is None:
                self.basis[pivot]=(row,rhs)
                return
            row^=old[0];rhs^=old[1]
        if rhs:
            raise ValueError('Inconsistent RNG observations or stream alignment')

    def next_word(self):
        if self.index==624:
            # In-place twist, matching CPython including the wrapped references.
            for i in range(624):
                joined=self.state[(i+1)%624][:31]+[self.state[i][31]]
                mixed=right(joined,1)
                mixed=[b^(joined[0] if (0x9908b0df>>j)&1 else 0) for j,b in enumerate(mixed)]
                self.state[i]=xor(self.state[(i+397)%624],mixed)
            self.index=0
        y=self.state[self.index][:];self.index+=1;self.words+=1
        y=xor(y,right(y,11))
        y=xor(y,left_mask(y,7,0x9d2c5680))
        y=xor(y,left_mask(y,15,0xefc60000))
        return xor(y,right(y,18))

    def observe_word(self,value=0,mask=0):
        """mask=0 records a skipped RNG draw without constraining its value."""
        word=self.next_word()
        for bit in range(32):
            if (mask>>bit)&1:
                self.equation(word[bit],(value>>bit)&1)

    def observe_random(self,value):
        n=int(value*2**53)
        self.observe_word((n>>26)<<5,0xffffffe0)
        self.observe_word((n&((1<<26)-1))<<6,0xffffffc0)

    def observe_uniform(self,a,b,value,error):
        """Conservative interval constraint for a rounded uniform(a,b) value."""
        if not b>a or error<0:
            raise ValueError('Need b>a and nonnegative coordinate uncertainty')
        lo=max(0,math.floor(((value-error-a)/(b-a))*2**53)-2)
        hi=min(2**53-1,math.ceil(((value+error-a)/(b-a))*2**53)+2)
        if lo>hi:
            raise ValueError('Uniform observation outside bounds')
        uncertain=(lo^hi).bit_length()
        mask=((1<<53)-1)^((1<<uncertain)-1)
        self.observe_word((lo>>26)<<5,(mask>>26)<<5)
        self.observe_word((lo&((1<<26)-1))<<6,(mask&((1<<26)-1))<<6)

    def clone(self):
        # Free variables are assigned zero. This returns a consistent hypothesis;
        # withheld outputs MUST confirm predictive uniqueness before use.
        solution=0
        for pivot,(row,rhs) in sorted(self.basis.items()):
            bit=rhs^((row&solution).bit_count()&1)
            solution|=bit<<pivot
        initial=[(solution>>(word*32))&0xffffffff for word in range(624)]
        clone=random.Random();clone.setstate((3,tuple(initial+[0]),None))
        for _ in range(self.words):clone.getrandbits(32)
        return clone
