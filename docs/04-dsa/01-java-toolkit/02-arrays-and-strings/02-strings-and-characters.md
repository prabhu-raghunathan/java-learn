---
title: "Strings, Characters & StringBuilder"
sidebar_position: 2
---

:::tip[TL;DR]
`String` is **immutable**: every "change" creates a new string, so building a string with `+=` in a loop is O(n²). Read with `String` methods (`charAt`, `substring`, `indexOf`, `split`), build with **`StringBuilder`**, and treat characters as small integers (`c - 'a'`) for frequency arrays.
:::

:::tip[Run it yourself]
Every snippet runs in `jshell`. Output is in the trailing comments.
:::

:::info[Solving a problem?]
For a lookup by task ("count each letter", "drop the last char", "split into words") and worked solutions to the classic string problems, see the [String Problem Playbook](./03-string-problem-playbook.md).
:::

## Complexity at a Glance

`n` = length of the string, `m` = length of the pattern / other string, `k` = length of the piece produced.

| Operation | Time | Extra space | Note |
| --- | --- | --- | --- |
| `length()`, `charAt(i)`, `isEmpty()` | **O(1)** | O(1) | backed by an array |
| `substring(i, j)` | O(k) | O(k) | copies since Java 7u6 |
| `indexOf`, `contains`, `lastIndexOf`, `replace` | O(n · m) worst, ~O(n) typical | O(1) (`replace`: O(n)) | naive matching, no KMP |
| `startsWith(p)` / `endsWith(p)` | O(m) | O(1) | |
| `equals`, `equalsIgnoreCase` | O(n) (O(1) if lengths differ) | O(1) | |
| `compareTo` | O(min(n, m)) | O(1) | stops at the first difference |
| `hashCode()` | O(n) first call, **O(1) after** | O(1) | cached in the String |
| `toCharArray()`, `chars()`, `toUpperCase()`, `trim()`, `strip()` | O(n) | O(n) | new array / new String |
| `split(regex)` | O(n) (+ regex cost) | O(n) | |
| `String.join`, `String.valueOf(char[])`, `new String(char[])` | O(total length) | O(total length) | |
| `a + b` | O(n + m) | O(n + m) | **in a loop: O(n²) total** |
| `Integer.parseInt(s)`, `String.valueOf(int)` | O(digits) | O(digits) | |
| `StringBuilder.append(x)` | **O(1) amortised** (+ length of x) | grows the buffer | |
| `StringBuilder.insert(0, x)` / `deleteCharAt(0)` | O(n) | O(1) | shifts everything |
| `StringBuilder.deleteCharAt(len - 1)` / `setLength(len - 1)` | **O(1)** | O(1) | backtracking undo |
| `StringBuilder.reverse()` | O(n) | O(1) | in place |
| `StringBuilder.toString()` | O(n) | O(n) | copies the buffer |

Space note: a fixed-size frequency array (`int[26]`, `int[128]`) is **O(1) extra space**, because its size doesn't grow with n. Say "O(1), or O(alphabet size)" in an interview.

## Reading a String

```java showLineNumbers
String s = "Hello World";
s.length();                     // 11       (method, unlike arr.length)
s.charAt(4);                    // 'o'
s.substring(6);                 // "World"  (from 6 to the end)
s.substring(0, 5);              // "Hello"  [begin, end): end is EXCLUSIVE
s.indexOf('o');                 // 4        first occurrence
s.indexOf('o', 5);              // 7        search starting at index 5
s.lastIndexOf('o');             // 7
s.indexOf("World");             // 6
s.indexOf('z');                 // -1       not found
s.contains("lo W");             // true
s.startsWith("He");             // true
s.endsWith("ld");               // true
s.toLowerCase();                // "hello world"
s.toUpperCase();                // "HELLO WORLD"
"abcdef".substring(2, 2);       // ""       (empty, not an error)
```

:::info[substring cost]
Since Java 7u6, `substring` **copies** the characters, so it is O(k) for a substring of length k, not O(1). Calling it inside a loop over all positions makes the loop O(n²).
:::

## Iterating Over Characters

```java showLineNumbers
for (int i = 0; i < s.length(); i++) {
    char c = s.charAt(i);                       // index-based, no extra memory
}

// highlight-next-line
for (char c : s.toCharArray()) { }              // copies into a char[] first (O(n) memory)

"hello".chars().filter(c -> c == 'l').count();  // 2   (chars() is an IntStream)
```

`toCharArray()` is also how you **modify** a string: change the `char[]`, then turn it back.

```java showLineNumbers
char[] ch = "hello".toCharArray();
ch[0] = 'j';
new String(ch);                 // "jello"
String.valueOf(ch);             // "jello"
String.valueOf(ch, 1, 3);       // "ell"  (offset 1, count 3)
```

## Splitting, Joining, Trimming

`split` takes a **regular expression**.

```java showLineNumbers
"a,b,,c".split(",");                    // [a, b, , c]   (empty middle piece kept)
"a,b,,".split(",");                     // [a, b]        trailing empties DROPPED
"a,b,,".split(",", -1);                 // [a, b, , ]    limit -1 keeps them
"a  b   c".trim().split("\\s+");        // [a, b, c]     any run of whitespace
// error-next-line
"a.b".split(".");                       // ❌ []  ('.' means "any char" in regex)
// success-next-line
"a.b".split("\\.");                     // ✅ [a, b]

String.join("-", List.of("a", "b", "c"));   // "a-b-c"
String.join("", new String[]{"x", "y"});    // "xy"

"  hi  ".trim();                        // "hi"
"  hi  ".strip();                       // "hi"  (Unicode-aware version, Java 11)
"".isEmpty();                           // true
"   ".isBlank();                        // true  (Java 11)
```

Other regex-special characters that need `\\` in `split`: `| . * + ? ( ) [ ] { } ^ $ \`.

## Replacing, Repeating, Formatting

```java showLineNumbers
"banana".replace('a', 'o');             // "bonono"   (all occurrences, not regex)
"banana".replace("an", "AN");           // "bANANa"
"a1b2".replaceAll("[0-9]", "");         // "ab"       (regex)
"ab".repeat(3);                         // "ababab"   (Java 11)
"Mississippi".matches("[A-Za-z]+");     // true       (whole string must match)
String.format("%d-%s-%.2f", 7, "x", 3.14159);   // "7-x-3.14"
String.format("%5d|%-5d|%05d", 42, 42, 42);     // "   42|42   |00042"
```

## Comparing Strings

```java showLineNumbers
String a1 = "java", a2 = new String("java");
// error-next-line
System.out.println(a1 == a2);           // ❌ false  (different objects)
// success-next-line
System.out.println(a1.equals(a2));      // ✅ true   (same characters)
"Abc".equalsIgnoreCase("aBC");          // true

"apple".compareTo("banana");            // -1   (negative: apple comes first)
"b".compareTo("a");                     // 1
"app".compareTo("apple");               // -2   (prefix is smaller; returns length difference)
"a".compareToIgnoreCase("B");           // -1
```

`compareTo` returns **negative / zero / positive**, not exactly -1 / 0 / 1. Always test `< 0`, `== 0`, `> 0`.

:::danger[Never compare strings with ==]
`==` sometimes "works" because string literals are pooled (interned), so tests with literals pass and then fail on strings read from input or built at runtime. Use `equals`. For null-safety use `Objects.equals(a, b)`.
:::

## Characters Are Numbers

A `char` is a 16-bit unsigned integer, so you can do arithmetic on it.

| Expression | Result | Use |
| --- | --- | --- |
| `'c' - 'a'` | `2` | letter → index 0..25 |
| `(char) ('a' + 2)` | `'c'` | index → letter |
| `'7' - '0'` | `7` | digit char → int |
| `(char) (3 + '0')` | `'3'` | int → digit char |
| `(int) 'A'`, `(int) 'a'`, `(int) '0'` | `65`, `97`, `48` | ASCII codes worth knowing |

```java showLineNumbers
int[] freq = new int[26];               // the classic letter counter
for (char c : "banana".toCharArray()) freq[c - 'a']++;
freq[0];                                // 3  ('a')
freq['b' - 'a'];                        // 1
freq['n' - 'a'];                        // 2
```

Use `int[128]` for any ASCII character, or `int[256]` for extended ASCII. Two strings are **anagrams** if their frequency arrays are equal (`Arrays.equals(f1, f2)`).

:::warning[char + char is an int]
```java showLineNumbers
// error-next-line
System.out.println('a' + 'b');          // ❌ 195 (97 + 98, numeric addition)
System.out.println("" + 'a' + 'b');     // ✅ ab  (a String on the left makes it concatenation)
System.out.println((char) ('a' + 1));   // b
```
:::

### The Character class

```java showLineNumbers
Character.isDigit('7');                 // true
Character.isLetter('x');                // true
Character.isLetterOrDigit('_');         // false  (handy for "valid palindrome" problems)
Character.isAlphabetic('Z');            // true
Character.isUpperCase('Q');             // true
Character.isLowerCase('Q');             // false
Character.isWhitespace(' ');            // true
Character.toUpperCase('a');             // 'A'
Character.toLowerCase('B');             // 'b'
Character.getNumericValue('9');         // 9
Character.getNumericValue('a');         // 10  (hex-style value, careful)
```

## Converting Between Strings and Numbers

```java showLineNumbers
Integer.parseInt("-123");               // -123   (returns int)
Integer.valueOf("7");                   // 7      (returns Integer)
Long.parseLong("9000000000");           // 9000000000
Double.parseDouble("2.5");              // 2.5
String.valueOf(42);                     // "42"
Integer.toString(42);                   // "42"
"" + 3.5;                               // "3.5"  (quick, a bit wasteful)

Integer.parseInt("101", 2);             // 5      (parse in base 2)
Integer.toString(5, 2);                 // "101"  (print in base 2)
Integer.toBinaryString(5);              // "101"
Integer.toHexString(255);               // "ff"
Integer.parseInt("ff", 16);             // 255

// error-next-line
Integer.parseInt("12a");                // ❌ NumberFormatException: For input string: "12a"
```

## StringBuilder: the Mutable String

Use it whenever you build a string piece by piece. `append` is amortised O(1).

```java showLineNumbers
StringBuilder sb = new StringBuilder();
sb.append("abc").append(1).append('x').append(2.5);   // chainable, takes any type
sb.toString();                          // "abc1x2.5"
sb.length();                            // 8
```

| Method | Example on `"hello"` | Result | Time |
| --- | --- | --- | --- |
| `insert(i, x)` | `insert(0, "[")` | `"[hello"` | O(n) (shifts the tail) |
| `reverse()` | `reverse()` | `"olleh"` (in place) | O(n) |
| `deleteCharAt(i)` | `deleteCharAt(0)` | `"ello"` | O(n − i) (O(1) at the end) |
| `delete(start, end)` | `delete(1, 3)` | `"hlo"` | O(n) |
| `setCharAt(i, c)` | `setCharAt(0, 'J')` | `"Jello"` | O(1) |
| `replace(start, end, str)` | `replace(0, 1, "HE")` | `"HEello"` | O(n) |
| `setLength(n)` | `setLength(length() - 1)` | `"hell"` (drop last char) | O(1) when shrinking |
| `charAt(i)` / `indexOf(s)` | `indexOf("l")` | `2` | O(1) / O(n · m) |

`new StringBuilder()` starts with room for **16** chars and grows to `2 × old + 2` when full (appending 17 chars → capacity 34). If you know the final size, `new StringBuilder(n)` avoids the regrowth copies.

The `setLength(len - 1)` / `deleteCharAt(len - 1)` pair is the standard **"undo"** step in backtracking:

```java showLineNumbers
void backtrack(StringBuilder path, ...) {
    path.append(c);                     // choose
    backtrack(path, ...);               // explore
    // highlight-next-line
    path.deleteCharAt(path.length() - 1);   // un-choose
}
```

### Palindrome check

```java showLineNumbers
new StringBuilder("racecar").reverse().toString().equals("racecar");   // true
```

For interviews, the **two-pointer** version is expected (O(1) extra space):

```java showLineNumbers
boolean isPalindrome(String s) {
    int i = 0, j = s.length() - 1;
    while (i < j) {
        if (s.charAt(i++) != s.charAt(j--)) return false;
    }
    return true;
}
```

:::danger[StringBuilder.equals compares identity]
```java showLineNumbers
StringBuilder e1 = new StringBuilder("a"), e2 = new StringBuilder("a");
// error-next-line
e1.equals(e2);                          // ❌ false (StringBuilder does not override equals)
e1.toString().equals(e2.toString());    // ✅ true
e1.compareTo(e2);                       // 0  (compareTo exists since Java 11)
```
Never use a `StringBuilder` as a `HashMap` key; call `toString()` first.
:::

## String vs StringBuilder vs StringBuffer

| | `String` | `StringBuilder` | `StringBuffer` |
| --- | --- | --- | --- |
| Mutable | No | Yes | Yes |
| Thread-safe | Yes (immutable) | No | Yes (synchronized) |
| Speed for building | O(n²) with `+=` in a loop | **Fastest** | Slower (locking) |
| Use in DSA | Keys, comparisons, input | **Building output, backtracking paths** | Practically never |

```java title="❌ O(n²): a new String every iteration" showLineNumbers
String out = "";
// error-next-line
for (int i = 0; i < n; i++) out += i;
```

```java title="✅ O(n)" showLineNumbers
StringBuilder out = new StringBuilder();
// success-next-line
for (int i = 0; i < n; i++) out.append(i);
```

## Anagram Key Pattern

Sorting the characters gives every anagram the same key (used in "Group Anagrams").

```java showLineNumbers
char[] key = "eat".toCharArray();
Arrays.sort(key);
String k = String.valueOf(key);         // "aet"  ("tea" and "ate" give the same)
```

| Key | Time per word of length k | Space per key |
| --- | --- | --- |
| sorted characters | O(k log k) | O(k) |
| frequency count (`Arrays.toString(freq)` or a `char[26]` turned into a String) | **O(k)** (+ O(26) to build the key) | O(26) |

For n words: sorted keys give O(n · k log k) overall, counted keys O(n · k).

## String Techniques Worth Knowing

| Technique | Time | Space | Example problem |
| --- | --- | --- | --- |
| Frequency array `int[26]` / `int[128]` | O(n) | O(1) | Valid Anagram, Ransom Note |
| Two pointers from both ends | O(n) | O(1) | Valid Palindrome |
| Sliding window + last-seen index | O(n) | O(1) (alphabet) | Longest Substring Without Repeating Characters |
| Expand around center | O(n²) | O(1) | Longest Palindromic Substring |
| `StringBuilder` + backtracking | O(branches^depth) | O(depth) | Generate Parentheses, Letter Combinations |
| KMP / Z-algorithm / rolling hash | O(n + m) | O(m) | Find the Index of the First Occurrence (when O(n · m) is too slow) |

### Anagram check in O(n) time, O(1) space

```java showLineNumbers
static boolean anagram(String a, String b) {
    if (a.length() != b.length()) return false;
    int[] f = new int[26];
    for (int i = 0; i < a.length(); i++) {
        f[a.charAt(i) - 'a']++;                            // count up for a
        f[b.charAt(i) - 'a']--;                            // count down for b
    }
    for (int x : f) if (x != 0) return false;
    return true;
}
// anagram("listen", "silent") → true,  anagram("rat", "car") → false
```

### Sliding window: longest substring without repeating characters

```java showLineNumbers
static int longest(String s) {
    int[] last = new int[128];                             // last index of each ASCII char
    Arrays.fill(last, -1);
    int best = 0;
    for (int l = 0, r = 0; r < s.length(); r++) {
        char c = s.charAt(r);
        // highlight-next-line
        if (last[c] >= l) l = last[c] + 1;                 // jump past the previous copy
        last[c] = r;
        best = Math.max(best, r - l + 1);
    }
    return best;
}
// longest("abcabcbb") → 3,  longest("pwwkew") → 3    O(n) time, O(1) space
```

### Expand around center: longest palindromic substring

```java showLineNumbers
static String longestPalindrome(String s) {
    int bestStart = 0, bestLen = 0;
    for (int c = 0; c < s.length(); c++) {
        for (int[] lr : new int[][]{{c, c}, {c, c + 1}}) {     // odd and even centers
            int l = lr[0], r = lr[1];
            while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) { l--; r++; }
            if (r - l - 1 > bestLen) { bestLen = r - l - 1; bestStart = l + 1; }
        }
    }
    return s.substring(bestStart, bestStart + bestLen);
}
// longestPalindrome("babad") → "bab",  longestPalindrome("cbbd") → "bb"    O(n²) time, O(1) space
```

## Memory Details Worth Knowing

- **Compact strings (Java 9+):** a `String` whose characters all fit in Latin-1 stores **1 byte per char**; otherwise 2 bytes per char (UTF-16). Still O(n) space either way.
- **String pool:** literals and compile-time constants are interned, so `"hello" == "hel" + "lo"` is `true`, `"hello" == new String("hello")` is `false`, and `new String("hello").intern() == "hello"` is `true`. Never rely on this in algorithms; use `equals`.
- Immutability means every "modification" allocates a new string: O(n) time **and** O(n) space each time.

## Gotchas

- Strings are **immutable**: `s.toUpperCase();` on its own line does nothing. Assign it: `s = s.toUpperCase();`
- `s.charAt(i)` returns a `char`, `s.substring(i, i + 1)` returns a `String`. `'a'` and `"a"` are not equal types; `"a".equals('a')` is `false`.
- `String.valueOf(char[])` gives the contents; `charArray.toString()` gives `[C@...`.
- `Character.isDigit` also accepts non-ASCII digits (like Arabic-Indic `'٣'`). For strict ASCII use `c >= '0' && c <= '9'`.
- `"abc".hashCode()` is computed from the characters (`96354`) and cached, so `String` keys hash fast after the first time.
