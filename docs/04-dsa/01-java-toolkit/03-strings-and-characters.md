---
title: "Strings, Characters & StringBuilder"
sidebar_position: 3
---

:::tip[TL;DR]
`String` is **immutable**: every "change" creates a new string, so building a string with `+=` in a loop is O(n²). Read with `String` methods (`charAt`, `substring`, `indexOf`, `split`), build with **`StringBuilder`**, and treat characters as small integers (`c - 'a'`) for frequency arrays.
:::

:::tip[Run it yourself]
Every snippet runs in `jshell`. Output is in the trailing comments.
:::

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

| Method | Example on `"hello"` | Result |
| --- | --- | --- |
| `insert(i, x)` | `insert(0, "[")` | `"[hello"` |
| `reverse()` | `reverse()` | `"olleh"` (in place, O(n)) |
| `deleteCharAt(i)` | `deleteCharAt(0)` | `"ello"` |
| `delete(start, end)` | `delete(1, 3)` | `"hlo"` |
| `setCharAt(i, c)` | `setCharAt(0, 'J')` | `"Jello"` |
| `replace(start, end, str)` | `replace(0, 1, "HE")` | `"HEello"` |
| `setLength(n)` | `setLength(length() - 1)` | `"hell"` (drop last char, O(1)) |
| `charAt(i)` / `indexOf(s)` | `indexOf("l")` | `2` |

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

For long strings, a frequency-count key (`Arrays.toString(freq)`) is O(n) instead of O(n log n).

## Gotchas

- Strings are **immutable**: `s.toUpperCase();` on its own line does nothing. Assign it: `s = s.toUpperCase();`
- `s.charAt(i)` returns a `char`, `s.substring(i, i + 1)` returns a `String`. `'a'` and `"a"` are not equal types; `"a".equals('a')` is `false`.
- `String.valueOf(char[])` gives the contents; `charArray.toString()` gives `[C@...`.
- `Character.isDigit` also accepts non-ASCII digits (like Arabic-Indic `'٣'`). For strict ASCII use `c >= '0' && c <= '9'`.
- `"abc".hashCode()` is computed from the characters (`96354`) and cached, so `String` keys hash fast after the first time.
