---
title: "Math, Overflow, Modulo & Bit Manipulation"
sidebar_position: 10
---

:::tip[TL;DR]
Most "wrong answer on a big test case" bugs in Java are **number bugs**: `int` overflow, integer division, negative `%`, or `Math.pow` returning a `double`. Know the limits (`int` ≈ ±2.1 × 10⁹, `long` ≈ ±9.2 × 10¹⁸), cast to `long` **before** multiplying, take `% MOD` at every step, and keep the `Integer` bit helpers (`bitCount`, `numberOfTrailingZeros`, ...) at hand.
:::

:::tip[Run it yourself]
Everything runs in `jshell`. `BigInteger`/`BigDecimal` need `import java.math.*;`.
:::

## Number Types and Limits

| Type | Bits | Range | Literal |
| --- | --- | --- | --- |
| `int` | 32 | −2,147,483,648 … 2,147,483,647 (≈ ±2.1 × 10⁹) | `42`, `1_000_000` |
| `long` | 64 | ≈ ±9.2 × 10¹⁸ | `42L` |
| `double` | 64 | ≈ ±1.8 × 10³⁰⁸, ~15–16 significant digits | `2.5`, `1e9` |
| `char` | 16 (unsigned) | 0 … 65,535 | `'a'` |
| `boolean` | | `true` / `false` | |

```java showLineNumbers
Integer.MAX_VALUE;              // 2147483647
Integer.MIN_VALUE;              // -2147483648
Long.MAX_VALUE;                 // 9223372036854775807
0b1010;                         // 10   binary literal
0x1F;                           // 31   hex literal
(int) 1e9;                      // 1000000000  (1e9 is a double; cast it)
```

**Rule of thumb:** if values or answers can exceed ~2 × 10⁹ (sums of large arrays, products, counts of pairs, factorials), use `long`.

## The Math Class

```java showLineNumbers
Math.max(3, 7);                 // 7
Math.min(3, 7);                 // 3
Math.abs(-5);                   // 5
Math.pow(2, 3);                 // 8.0   ← returns a DOUBLE
(int) Math.pow(2, 3);           // 8
Math.sqrt(16);                  // 4.0
(int) Math.sqrt(26);            // 5     (floor of the square root)
Math.cbrt(27);                  // 3.0
Math.hypot(3, 4);               // 5.0   (sqrt(x² + y²), distance)
Math.log(Math.E);               // 1.0   (natural log)
Math.log10(1000);               // 3.0
Math.log(8) / Math.log(2);      // 3.0   (log base 2)
Math.signum(-4.2);              // -1.0
```

### Rounding

| Call | Result | Note |
| --- | --- | --- |
| `Math.ceil(7 / 2)` | `3.0` | ❌ `7 / 2` is already the integer `3` |
| `Math.ceil(7 / 2.0)` | `4.0` | ✅ make one side a double |
| `(7 + 2 - 1) / 2` | `4` | ✅ integer ceiling: `(a + b - 1) / b` for positive values |
| `Math.ceilDiv(7, 2)` <Since v="18" /> | `4` | built-in integer ceiling |
| `Math.floor(-2.5)` | `-3.0` | toward −∞ |
| `Math.round(2.5)` / `Math.round(-2.5)` | `3` / `-2` | rounds half **up** (toward +∞) |

## Integer Division and the Sign of %

```java showLineNumbers
5 / 2;                          // 2      (integer division truncates)
5 / 2.0;                        // 2.5
(double) 5 / 2;                 // 2.5    (cast BEFORE dividing)
(double) (5 / 2);               // 2.0    ❌ divided first, then cast
-7 / 2;                         // -3     truncates toward zero
-7 % 3;                         // -1     ← % keeps the sign of the LEFT operand
Math.floorDiv(-7, 2);           // -4     floors toward -∞
Math.floorMod(-7, 3);           // 2      always non-negative for a positive divisor
((-7 % 3) + 3) % 3;             // 2      same thing by hand
```

Negative `%` bites in circular arrays (`(i - 1) % n` when `i = 0`) and hashing. Use `Math.floorMod(i - 1, n)` or `(i - 1 + n) % n`.

## Overflow

Java integer arithmetic **wraps around silently**: no exception, just a wrong number.

```java showLineNumbers
Integer.MAX_VALUE + 1;          // -2147483648

int a = 100000, b = 100000;
// error-next-line
long bad = a * b;               // ❌ 1410065408  (multiplied as int, THEN widened)
// error-next-line
long bad2 = (long) (a * b);     // ❌ 1410065408  (same: cast happens too late)
// success-next-line
long good = (long) a * b;       // ✅ 10000000000 (a becomes long first)
```

```java showLineNumbers
Math.addExact(Integer.MAX_VALUE, 1);    // ArithmeticException: integer overflow
Math.multiplyExact(a, b);               // ArithmeticException: integer overflow
// error-next-line
Math.abs(Integer.MIN_VALUE);            // ❌ -2147483648 (no positive counterpart exists)
```

### The binary-search midpoint

```java showLineNumbers
int lo = 2_000_000_000, hi = 2_100_000_000;
// error-next-line
(lo + hi) / 2;                  // ❌ -97483648   (lo + hi overflowed)
// success-start
lo + (hi - lo) / 2;             // ✅ 2050000000
(lo + hi) >>> 1;                // ✅ 2050000000  (unsigned shift reads the overflowed bits correctly)
// success-end
```

### Choosing "infinity"

```java showLineNumbers
int INF = Integer.MAX_VALUE;
// error-next-line
INF + 1;                        // ❌ -2147483648: dist[u] + w overflows in Dijkstra/DP

int INF2 = (int) 1e9;           // ✅ common choice
INF2 + INF2;                    // 2000000000  (still fits in an int)
```

If you do use `Integer.MAX_VALUE`, check `dist[u] != Integer.MAX_VALUE` before adding to it.

### Math.pow loses precision on big integers

```java showLineNumbers
(long) Math.pow(3, 38);         // 1350851717672992000   ❌ last digits wrong (double has ~16 digits)
long p = 1;
for (int i = 0; i < 38; i++) p *= 3;    // 1350851717672992089   ✅ exact
```

Use `Math.pow` for small exponents or real-number answers; for exact integer powers, multiply in a loop or use fast exponentiation (below).

## Modular Arithmetic (10⁹ + 7)

"Return the answer modulo 10⁹ + 7" means: keep every intermediate value reduced so it never overflows.

```java showLineNumbers
static final int MOD = 1_000_000_007;

long s = 0;
for (int i = 0; i < 5; i++) s = (s + 1_000_000_000) % MOD;   // 999999972
long prod = (long) x * y % MOD;                              // cast before multiplying!
long diff = ((a - b) % MOD + MOD) % MOD;                     // keep subtraction non-negative
```

| Operation | Safe form |
| --- | --- |
| add | `(a + b) % MOD` |
| subtract | `((a - b) % MOD + MOD) % MOD` |
| multiply | `(long) a * b % MOD` |
| divide by `b` | multiply by the inverse: `a * power(b, MOD - 2, MOD) % MOD` (MOD must be prime) |

### Fast power (binary exponentiation), O(log e)

```java title="FastPower.java" showLineNumbers
static long power(long b, long e, long m) {
    long r = 1;
    b %= m;
    while (e > 0) {
        // highlight-next-line
        if ((e & 1) == 1) r = r * b % m;       // this bit of e is set: multiply it in
        b = b * b % m;                         // square for the next bit
        e >>= 1;
    }
    return r;
}
// power(2, 10, MOD) = 1024,  power(2, 100, MOD) = 976371285
// power(3, MOD - 2, MOD) * 3 % MOD = 1   (modular inverse of 3)
```

### GCD, LCM, primes

```java showLineNumbers
static long gcd(long a, long b) { return b == 0 ? a : gcd(b, a % b); }    // Euclid
gcd(12, 18);                                   // 6
long lcm = 12 / gcd(12, 18) * 18;              // 36  (divide first to avoid overflow)
BigInteger.valueOf(12).gcd(BigInteger.valueOf(18));   // 6  (built-in)

static boolean isPrime(int n) {                // O(√n)
    if (n < 2) return false;
    for (int i = 2; (long) i * i <= n; i++)    // i * i as long: no overflow near MAX_VALUE
        if (n % i == 0) return false;
    return true;
}
// isPrime(97) → true, isPrime(1) → false
```

```java title="Sieve.java" showLineNumbers
static int[] sieve(int n) {                    // all primes ≤ n in O(n log log n)
    boolean[] comp = new boolean[n + 1];
    List<Integer> pr = new ArrayList<>();
    for (int i = 2; i <= n; i++) {
        if (!comp[i]) {
            pr.add(i);
            for (long j = (long) i * i; j <= n; j += i) comp[(int) j] = true;
        }
    }
    return pr.stream().mapToInt(Integer::intValue).toArray();
}
// sieve(20) → [2, 3, 5, 7, 11, 13, 17, 19]
```

## BigInteger & BigDecimal

Arbitrary precision, immutable, slow. Use them when even `long` overflows (factorials, huge Fibonacci, "add two numbers as strings" follow-ups) or when you need exact decimals.

```java showLineNumbers
BigInteger bi = new BigInteger("123456789012345678901234567890");
bi.multiply(BigInteger.TWO);                   // 246913578024691357802469135780
bi.mod(BigInteger.valueOf(97));                // 52
BigInteger.valueOf(2).pow(64);                 // 18446744073709551616
BigInteger.ONE.shiftLeft(100);                 // 1267650600228229401496703205376
bi.compareTo(BigInteger.ZERO);                 // 1   (no < or > on objects)
BigInteger.valueOf(17).isProbablePrime(10);    // true

BigInteger f = BigInteger.ONE;
for (int i = 2; i <= 25; i++) f = f.multiply(BigInteger.valueOf(i));
// 25! = 15511210043330985984000000

0.1 + 0.2;                                     // 0.30000000000000004
0.1 + 0.2 == 0.3;                              // false   ← never == on doubles
Math.abs(0.1 + 0.2 - 0.3) < 1e-9;              // true    compare with a tolerance
new BigDecimal("0.1").add(new BigDecimal("0.2"));   // 0.3  (exact; build from Strings)
```

Other methods: `add`, `subtract`, `divide`, `modPow`, `modInverse`, `sqrt` (Java 9), `toString(radix)`, `longValue()`, `intValueExact()`.

## Bit Manipulation

### Operators

| Operator | Name | Example (`n = 12` = `1100₂`) |
| --- | --- | --- |
| `&` | AND | `n & 1` → `0` (even) |
| `\|` | OR | `n \| 1` → `13` |
| `^` | XOR | `n ^ 8` → `4` |
| `~` | NOT | `~n` → `-13` |
| `<<` | left shift (× 2ᵏ) | `n << 1` → `24` |
| `>>` | right shift, keeps sign (÷ 2ᵏ, floors) | `n >> 1` → `6`, `-16 >> 2` → `-4` |
| `>>>` | right shift, fills with 0 | `-16 >>> 28` → `15` |

### The tricks you actually use

| Goal | Code | `n = 12` (`1100₂`) |
| --- | --- | --- |
| Is bit `i` set? | `(n >> i) & 1` | `i = 2` → `1` |
| Set bit `i` | `n \| (1 << i)` | `i = 0` → `13` |
| Clear bit `i` | `n & ~(1 << i)` | `i = 2` → `8` |
| Toggle bit `i` | `n ^ (1 << i)` | `i = 3` → `4` |
| Drop the lowest set bit | `n & (n - 1)` | `8` |
| Keep only the lowest set bit | `n & -n` | `4` |
| Power of two? | `n > 0 && (n & (n - 1)) == 0` | `false` (16 → `true`) |
| Odd? | `(n & 1) == 1` | `false` |

### Integer / Long helpers

```java showLineNumbers
Integer.bitCount(13);                   // 3    number of 1-bits (popcount)
Integer.toBinaryString(13);             // "1101"
Integer.highestOneBit(13);              // 8
Integer.lowestOneBit(12);               // 4
Integer.numberOfTrailingZeros(12);      // 2    index of the lowest set bit
Integer.numberOfLeadingZeros(1);        // 31
31 - Integer.numberOfLeadingZeros(13);  // 3    floor(log2(13))
Integer.reverse(1);                     // -2147483648 (bit 0 moved to bit 31)
Long.bitCount(255L);                    // 8
Integer.toBinaryString(-1).length();    // 32   (two's complement: all ones)
```

### XOR finds the single number

`x ^ x = 0` and `x ^ 0 = x`, so pairs cancel out:

```java showLineNumbers
int[] arr = {4, 1, 2, 1, 2};
int xr = 0;
for (int v : arr) xr ^= v;              // 4
```

### Enumerate all subsets with a bitmask

Each number from `0` to `2ⁿ − 1` is one subset: bit `i` says whether element `i` is in it.

```java showLineNumbers
int[] set = {1, 2, 3};
List<List<Integer>> subs = new ArrayList<>();
for (int mask = 0; mask < (1 << set.length); mask++) {
    List<Integer> cur = new ArrayList<>();
    for (int i = 0; i < set.length; i++)
        // highlight-next-line
        if ((mask >> i & 1) == 1) cur.add(set[i]);
    subs.add(cur);
}
// [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
```

:::danger[Shifting an int by 40 does not give 2⁴⁰]
`int` shifts only use the low 5 bits of the distance, so `1 << 40` is `1 << 8` = `256`, and `1 << 31` is already negative (`-2147483648`). For big masks use a long: `1L << 40` → `1099511627776`.
:::

## Gotchas

- `Integer` objects from collections: `==` works only for -128..127 (`Integer.valueOf(127) == Integer.valueOf(127)` is `true`, `128` is `false`). Use `equals` or unbox first.
- `char` arithmetic gives an `int`: `'a' + 1` is `98`; cast back with `(char)`.
- `Math.round` returns a `long` for a `double` argument (and an `int` for a `float`), so `int r = Math.round(2.5);` doesn't compile without a cast.
- `new Random(seed)` gives reproducible random numbers (`rnd.nextInt(n)` → `0..n-1`); `ThreadLocalRandom.current().nextInt(a, b)` is the quick unseeded version.
