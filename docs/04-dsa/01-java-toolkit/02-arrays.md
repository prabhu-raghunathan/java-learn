---
title: "Arrays & the Arrays Utility Class"
sidebar_position: 2
---

:::tip[TL;DR]
Arrays are the fastest structure Java has: fixed size, contiguous memory, O(1) index access, no boxing. The `java.util.Arrays` class gives you everything else: `fill`, `sort`, `binarySearch`, `copyOf`, `copyOfRange`, `equals`, `toString`, `deepToString`, `asList` and `stream`.
:::

:::tip[Run it yourself]
Every snippet on this page runs as-is in `jshell` (type `jshell` in a terminal; `java.util.*` is imported by default there).
:::

## Creating Arrays

```java showLineNumbers
int[] a = new int[5];                      // [0, 0, 0, 0, 0]
int[] arr = {5, 3, 8, 1};                  // literal; arr.length == 4 (a field, no brackets)
int[][] grid = new int[3][4];              // 3 rows, 4 cols: grid.length == 3, grid[0].length == 4
int[][] jagged = new int[3][];             // rows created later, each can have its own length
jagged[0] = new int[]{1};
jagged[1] = new int[]{1, 2};
jagged[2] = new int[0];
System.out.println(Arrays.deepToString(jagged));      // [[1], [1, 2], []]
```

### Default values

| Element type | Default |
| --- | --- |
| `int`, `long`, `short`, `byte` | `0` |
| `double`, `float` | `0.0` |
| `boolean` | `false` |
| `char` | `'\u0000'` (character code 0, prints as nothing) |
| any object (`String`, `Integer`, `int[]`) | `null` |

:::warning[Integer[] defaults to null]
`Integer[] memo = new Integer[n];` starts as all `null`, not `0`. That is actually useful for memoisation ("`null` = not computed yet"), but `memo[i] + 1` on an unset cell throws `NullPointerException`.
:::

### Direction arrays for grids

Declare the 4 (or 8) neighbour offsets once instead of writing four `if` blocks.

```java showLineNumbers
int[] dr = {-1, 1, 0, 0};
int[] dc = {0, 0, -1, 1};
for (int d = 0; d < 4; d++) {
    int nr = r + dr[d], nc = c + dc[d];
    // highlight-next-line
    if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) continue;   // bounds check first
    // visit (nr, nc)
}
```

## Printing

```java showLineNumbers
int[] a = {1, 2, 3};
// error-next-line
System.out.println(a);                         // ❌ something like [I@6d06d69c (type + hash, not contents)
System.out.println(Arrays.toString(a));        // ✅ [1, 2, 3]

int[][] g = {{1, 2}, {3, 4}};
System.out.println(Arrays.toString(g));        // ❌ [[I@7852e922, [I@4e25154f] (or similar)
System.out.println(Arrays.deepToString(g));    // ✅ [[1, 2], [3, 4]]
```

## Filling

```java showLineNumbers
int[] f = new int[5];
Arrays.fill(f, -1);                             // [-1, -1, -1, -1, -1]

long[] big = new long[2];
Arrays.fill(big, 1, 2, 7L);                     // fill index range [1, 2): [0, 7]

int[][] dp = new int[2][3];
// highlight-next-line
for (int[] row : dp) Arrays.fill(row, Integer.MAX_VALUE);   // fill works on ONE dimension
// [[2147483647, 2147483647, 2147483647], [2147483647, 2147483647, 2147483647]]

int[] sq = new int[5];
Arrays.setAll(sq, i -> i * i);                  // [0, 1, 4, 9, 16]  (value from the index)
```

:::danger[Arrays.fill with an object fills the SAME object everywhere]
`List<Integer>[] adj = new List[n]; Arrays.fill(adj, new ArrayList<>());` puts **one** list in every slot, so adding an edge to `adj[0]` shows up in all of them. Create each one in a loop: `for (int i = 0; i < n; i++) adj[i] = new ArrayList<>();`
:::

## Sorting

```java showLineNumbers
int[] so = {5, 3, 8, 1, 9, 2};
Arrays.sort(so);                  // [1, 2, 3, 5, 8, 9]

int[] part = {5, 3, 8, 1, 9, 2};
Arrays.sort(part, 1, 4);          // only [1, 4): [5, 1, 3, 8, 9, 2]

char[] chs = "dcba".toCharArray();
Arrays.sort(chs);                 // "abcd" (sorted chars = anagram key)

String[] words = {"pear", "fig", "apple"};
Arrays.sort(words);               // [apple, fig, pear]  alphabetical
// by length: [fig, pear, apple]
Arrays.sort(words, (p, q) -> p.length() - q.length());
```

### Sorting descending

A comparator only works on **object** arrays, so `int[]` can't be sorted descending directly.

```java title="❌ Comparator on a primitive array" showLineNumbers
int[] a = {3, 1};
// error-next-line
Arrays.sort(a, Collections.reverseOrder());
// error: no suitable method found for sort(int[],Comparator<Object>)
```

```java showLineNumbers
// success-start
// Option 1: box it
Integer[] boxed = {5, 3, 8, 1};
Arrays.sort(boxed, Collections.reverseOrder());  // [8, 5, 3, 1]

// Option 2: sort ascending, then reverse in place (fastest, no boxing)
int[] desc = {5, 3, 8, 1};
Arrays.sort(desc);
for (int i = 0, j = desc.length - 1; i < j; i++, j--) {
    int t = desc[i]; desc[i] = desc[j]; desc[j] = t;
}                                                // [8, 5, 3, 1]

// Option 3: stream (short, but slowest)
int[] desc2 = IntStream.of(5, 3, 8, 1).boxed()
        .sorted(Collections.reverseOrder())
        .mapToInt(Integer::intValue).toArray();  // [8, 5, 3, 1]
// success-end
```

### Sorting a 2D array (intervals, pairs)

`int[][]` is an array of `int[]` objects, so comparators **do** work on it.

```java showLineNumbers
int[][] intervals = {{3, 5}, {1, 4}, {2, 2}};
// highlight-next-line
Arrays.sort(intervals, (p, q) -> Integer.compare(p[0], q[0]));   // by start
System.out.println(Arrays.deepToString(intervals));              // [[1, 4], [2, 2], [3, 5]]
```

More comparator patterns (by end, then by start, descending, etc.) are on the [Sorting & Comparators](./09-sorting-and-comparators.md) page.

### What algorithm does Arrays.sort use?

| Call | Algorithm | Stable? | Worst case |
| --- | --- | --- | --- |
| `Arrays.sort(int[])` and other primitives | **Dual-Pivot Quicksort** | No (doesn't matter for primitives) | O(n log n) since JDK 14 (falls back to heapsort); O(n²) on older JDKs |
| `Arrays.sort(Object[])`, `Collections.sort`, `List.sort` | **TimSort** (merge sort + insertion sort) | **Yes** | O(n log n) |

"Stable" means equal elements keep their original order, which matters when you sort by one key and then another.

## Binary Search

The array **must already be sorted**.

```java showLineNumbers
int[] bs = {1, 3, 5, 7};
Arrays.binarySearch(bs, 5);                     // 2   (found: its index)
Arrays.binarySearch(bs, 4);                     // -3  (not found)
Arrays.binarySearch(bs, 9);                     // -5
Arrays.binarySearch(bs, 0);                     // -1
```

When the key is missing it returns **`-(insertionPoint) - 1`**, where the insertion point is where the key *would* go. Recover it like this:

```java showLineNumbers
int idx = Arrays.binarySearch(bs, 4);           // -3
// highlight-next-line
int insertAt = idx < 0 ? -idx - 1 : idx;        // 2  (4 would go before 5)
```

:::warning[Duplicates]
With duplicates, `binarySearch` returns **any** matching index, not necessarily the first or last. For "first index ≥ x" (lower bound) or "first index > x" (upper bound), write your own binary search, or use a `TreeMap`/`TreeSet` (`ceiling`, `higher`).
:::

## Copying

```java showLineNumbers
int[] src = {1, 2, 3, 4, 5};
Arrays.copyOf(src, 3);                          // [1, 2, 3]            (truncate)
Arrays.copyOf(src, 7);                          // [1, 2, 3, 4, 5, 0, 0] (pad with defaults)
Arrays.copyOfRange(src, 1, 4);                  // [2, 3, 4]            (from inclusive, to exclusive)
int[] cl = src.clone();                         // full copy

int[] dst = new int[5];
System.arraycopy(src, 0, dst, 1, 3);            // (src, srcPos, dest, destPos, length): [0, 1, 2, 3, 0]
```

:::danger[clone() on a 2D array is shallow]
`grid.clone()` copies the outer array only: both arrays share the same rows.

```java showLineNumbers
int[][] g = {{1, 2}, {3, 4}};
int[][] shallow = g.clone();
shallow[0][0] = 99;
// error-next-line
System.out.println(g[0][0]);                    // ❌ 99  (original changed too)

// success-start
int[][] deep = new int[g.length][];
for (int i = 0; i < g.length; i++) deep[i] = g[i].clone();   // ✅ copy each row
// success-end
```
:::

## Comparing & Hashing

```java showLineNumbers
int[] x = {1, 2}, y = {1, 2};
// error-next-line
x.equals(y);                                    // ❌ false (identity, like ==)
Arrays.equals(x, y);                            // ✅ true  (element by element)
Arrays.deepEquals(new int[][]{{1}}, new int[][]{{1}});   // true (for nested arrays)
Arrays.hashCode(x) == Arrays.hashCode(y);       // true  (content-based hash)
Arrays.mismatch(new int[]{1, 2, 3}, new int[]{1, 5, 3}); // 1  (first differing index, -1 if equal)
Arrays.compare(new int[]{1, 2}, new int[]{1, 3});        // -1 (lexicographic, like compareTo)
```

## Quick Stats with Streams

```java showLineNumbers
int[] st = {4, 2, 7};
Arrays.stream(st).sum();                        // 13
Arrays.stream(st).max().getAsInt();             // 7
Arrays.stream(st).min().getAsInt();             // 2
Arrays.stream(st).average().orElse(0);          // 4.333333333333333
Arrays.stream(new int[]{1, 2, 3, 4, 5}, 1, 3).sum();     // 5  (range [1, 3))
```

:::danger[IntStream.sum() overflows silently]
```java showLineNumbers
int[] v = {Integer.MAX_VALUE, 1};
// error-next-line
Arrays.stream(v).sum();                         // ❌ -2147483648
// success-next-line
Arrays.stream(v).asLongStream().sum();          // ✅ 2147483648
```
:::

### Prefix sums in one line

```java showLineNumbers
int[] pre = {1, 2, 3, 4};
Arrays.parallelPrefix(pre, Integer::sum);       // [1, 3, 6, 10]  (in place)
```

For interview code, a plain loop is clearer: `for (int i = 1; i < n; i++) pre[i] += pre[i - 1];`

## Converting Between Arrays and Lists

| From → To | Code |
| --- | --- |
| `int[]` → `List<Integer>` | `Arrays.stream(a).boxed().collect(Collectors.toList())` (or `.toList()` for read-only) |
| `List<Integer>` → `int[]` | `list.stream().mapToInt(Integer::intValue).toArray()` |
| `List<String>` → `String[]` | `list.toArray(new String[0])` |
| `String[]` / `Integer[]` → `List` | `Arrays.asList(arr)` (fixed-size view) or `new ArrayList<>(Arrays.asList(arr))` |
| `int[][]` from `List<int[]>` | `list.toArray(new int[0][])` (common in "merge intervals") |

### Arrays.asList traps

```java showLineNumbers
List<Integer> asl = Arrays.asList(3, 1, 2);
asl.set(0, 9);                                  // OK: [9, 1, 2]
// error-next-line
asl.add(4);                                     // ❌ UnsupportedOperationException (fixed size)

String[] sa = {"a", "b"};
List<String> view = Arrays.asList(sa);
sa[0] = "z";
System.out.println(view);                       // [z, b]  (it's a VIEW over the array)

int[] prim = {1, 2};
// error-next-line
Arrays.asList(prim).size();                     // ❌ 1  (a List<int[]> holding one array!)
```

## Gotchas

- `arr.length` (field) vs `str.length()` (method) vs `list.size()` (method). Mixing them up is the most common compile error under time pressure.
- `ArrayIndexOutOfBoundsException`: check `i >= 0 && i < n` **before** reading `arr[i]` (the `&&` short-circuits).
- Java has no negative indexing: `arr[-1]` throws. The last element is `arr[arr.length - 1]`.
- A generic array like `new List<Integer>[n]` does not compile; write `List<Integer>[] adj = new List[n];` (an unchecked warning, fine in DSA) or use `List<List<Integer>>`.
- Very large local arrays (like `new int[10^8]`) can exhaust the heap. `10^6` to `10^7` ints is normal (4 to 40 MB).
