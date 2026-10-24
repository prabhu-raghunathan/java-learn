---
title: "Arrays & the Arrays Utility Class"
sidebar_position: 1
---

:::tip[TL;DR]
Arrays are the fastest structure Java has: fixed size, contiguous memory, O(1) index access, no boxing. The `java.util.Arrays` class gives you everything else: `fill`, `sort`, `binarySearch`, `copyOf`, `copyOfRange`, `equals`, `toString`, `deepToString`, `asList` and `stream`.
:::

:::tip[Run it yourself]
Every snippet on this page runs as-is in `jshell` (type `jshell` in a terminal; `java.util.*` is imported by default there).
:::

## Complexity at a Glance

`n` = array length, `k` = length of the range or copy.

| Operation | Time | Extra space | Note |
| --- | --- | --- | --- |
| `new int[n]` | O(n) | O(n) | the JVM zeroes every slot |
| `new int[R][C]` | O(R · C) | O(R · C) | |
| `arr[i]` read / write | **O(1)** | O(1) | contiguous memory: address = base + i × size |
| `arr.length` | O(1) | O(1) | stored in the array header |
| `Arrays.fill(arr, x)` / `setAll` | O(n) | O(1) | |
| `Arrays.toString` / `deepToString` | O(n) / O(total cells) | O(n) | builds a String |
| `Arrays.sort(int[])` | O(n log n) | O(log n) typical | Dual-Pivot Quicksort |
| `Arrays.sort(Integer[] / String[] / int[][])` | O(n log n) comparisons | O(n) | TimSort; each comparison may cost more (e.g. O(L) for strings) |
| `Arrays.binarySearch` | **O(log n)** | O(1) | array must be sorted |
| `Arrays.copyOf` / `copyOfRange` / `clone()` / `System.arraycopy` | O(k) | O(k) (`arraycopy` O(1): it writes into an existing array) | |
| `Arrays.equals` / `Arrays.hashCode` / `mismatch` / `compare` | O(n) | O(1) | |
| `Arrays.stream(arr).sum()` / `max()` | O(n) | O(1) | |
| `Arrays.asList(arr)` | **O(1)** | O(1) | a view over the same array, no copy |
| `new ArrayList<>(Arrays.asList(arr))` / `stream().boxed()...` | O(n) | O(n) | copies and boxes |
| linear search (a loop) | O(n) | O(1) | |
| insert / delete at index i (by shifting) | O(n) | O(1) | arrays are fixed-size; that's what `ArrayList` does for you |

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

Allocation is **O(n) time and O(n) space**: Java always zero-fills a new array, so `new int[n]` inside a loop that runs n times is O(n²).

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

Each neighbour check is O(1), so visiting every cell of an R × C grid with this loop is **O(R · C) time**.

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

More comparator patterns (by end, then by start, descending, etc.) are on the [Sorting & Comparators](../04-sorting-math-and-io/01-sorting-and-comparators.md) page.

### What algorithm does Arrays.sort use?

| Call | Algorithm | Stable? | Worst case |
| --- | --- | --- | --- |
| `Arrays.sort(int[])` and other primitives | **Dual-Pivot Quicksort** | No (doesn't matter for primitives) | O(n log n) since JDK 14 (falls back to heapsort); O(n²) on older JDKs |
| `Arrays.sort(Object[])`, `Collections.sort`, `List.sort` | **TimSort** (merge sort + insertion sort) | **Yes** | O(n log n) |

"Stable" means equal elements keep their original order, which matters when you sort by one key and then another.

## Binary Search

The array **must already be sorted**. **O(log n) time, O(1) space**: each step halves the search range (10⁶ elements → at most 20 steps). Sorting first costs O(n log n), so binary search pays off only when you search many times or the data is already sorted.

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

Every copy is **O(k) time and O(k) space** for k copied elements. Copying inside a loop is a common hidden O(n²).

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

| From → To | Code | Cost |
| --- | --- | --- |
| `int[]` → `List<Integer>` | `Arrays.stream(a).boxed().collect(Collectors.toList())` (or `.toList()` for read-only) | O(n) time, O(n) space (boxes every value) |
| `List<Integer>` → `int[]` | `list.stream().mapToInt(Integer::intValue).toArray()` | O(n), O(n) |
| `List<String>` → `String[]` | `list.toArray(new String[0])` | O(n), O(n) |
| `String[]` / `Integer[]` → `List` | `Arrays.asList(arr)` (fixed-size view) or `new ArrayList<>(Arrays.asList(arr))` | view: **O(1)**; copy: O(n) |
| `int[][]` from `List<int[]>` | `list.toArray(new int[0][])` (common in "merge intervals") | O(n) (copies the row references, not the rows) |

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

## Array Techniques Worth Knowing

The building blocks behind many array problems, with their cost.

| Technique | Build | Query / update | Space | Use for |
| --- | --- | --- | --- | --- |
| Prefix sum | O(n) | range sum O(1) | O(n) | many range-sum queries, subarray sums |
| Difference array | O(1) per range update | rebuild O(n) | O(n) | many "add x to every element in [l, r]" updates |
| Reverse in place (two pointers) | | O(n) | **O(1)** | rotate, palindromes, reverse words |
| Rotate by k with three reversals | | O(n) | **O(1)** | Rotate Array (vs O(n) space with a copy) |
| Counting array `int[maxValue + 1]` | O(n + range) | O(1) per lookup | O(range) | values in a small known range (faster than a `HashMap`) |
| 2D prefix sum | O(R · C) | sub-rectangle sum O(1) | O(R · C) | Range Sum Query 2D |
| Transpose + reverse rows | | O(n²) for n × n | **O(1)** | Rotate Image 90° in place |

### Prefix sums

`pre[i]` = sum of the first `i` elements, so `sum(l..r) = pre[r + 1] - pre[l]`.

```java showLineNumbers
int[] v = {3, 1, 4, 1, 5, 9, 2};
int[] pre = new int[v.length + 1];                         // pre[0] = 0: the empty prefix
for (int i = 0; i < v.length; i++) pre[i + 1] = pre[i] + v[i];
// pre = [0, 3, 4, 8, 9, 14, 23, 25]
// highlight-next-line
int sum2to5 = pre[6] - pre[2];                             // 19  (4 + 1 + 5 + 9), O(1)
```

Use `long[]` for the prefix array if the sum can exceed ~2.1 × 10⁹.

### Difference array (range updates)

```java showLineNumbers
int n = 5;
int[] diff = new int[n + 1];
int[][] updates = {{1, 3, 10}, {2, 4, 5}};                 // add 10 to [1..3], add 5 to [2..4]
for (int[] u : updates) {
    diff[u[0]] += u[2];                                    // start adding here
    diff[u[1] + 1] -= u[2];                                // stop after r
}
int[] res = new int[n];
for (int i = 0, run = 0; i < n; i++) { run += diff[i]; res[i] = run; }
// res = [0, 10, 15, 15, 5]   (q updates + one pass: O(q + n) instead of O(q · n))
```

### Rotate right by k in O(1) extra space

```java showLineNumbers
static void reverse(int[] a, int i, int j) {
    while (i < j) { int t = a[i]; a[i++] = a[j]; a[j--] = t; }
}
static void rotateRight(int[] a, int k) {
    int n = a.length;
    // highlight-next-line
    k %= n;                                                // k can be larger than n
    reverse(a, 0, n - 1);                                  // 7 6 5 4 3 2 1
    reverse(a, 0, k - 1);                                  // 5 6 7 | 4 3 2 1
    reverse(a, k, n - 1);                                  // 5 6 7 | 1 2 3 4
}
// rotateRight({1, 2, 3, 4, 5, 6, 7}, 3) → [5, 6, 7, 1, 2, 3, 4]
```

### Rotate a square matrix 90° clockwise in place

Transpose (swap across the diagonal), then reverse each row.

```java showLineNumbers
int[][] m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int N = m.length;
for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++) { int t = m[i][j]; m[i][j] = m[j][i]; m[j][i] = t; }
for (int[] row : m) reverse(row, 0, N - 1);
// [[7, 4, 1], [8, 5, 2], [9, 6, 3]]   O(N²) time, O(1) extra space
```

### Counting array instead of a map

```java showLineNumbers
int[] v = {3, 1, 4, 1, 5, 9, 2};
int[] cnt = new int[10];                                   // values known to be 0..9
for (int q : v) cnt[q]++;
// cnt = [0, 2, 1, 1, 1, 1, 0, 0, 0, 1]
```

O(n + range) time and O(range) space, with no hashing or boxing. If the range is huge (values up to 10⁹), use a `HashMap` instead.

## Gotchas

- `arr.length` (field) vs `str.length()` (method) vs `list.size()` (method). Mixing them up is the most common compile error under time pressure.
- `ArrayIndexOutOfBoundsException`: check `i >= 0 && i < n` **before** reading `arr[i]` (the `&&` short-circuits).
- Java has no negative indexing: `arr[-1]` throws. The last element is `arr[arr.length - 1]`.
- A generic array like `new List<Integer>[n]` does not compile; write `List<Integer>[] adj = new List[n];` (an unchecked warning, fine in DSA) or use `List<List<Integer>>`.
- Very large local arrays (like `new int[10^8]`) can exhaust the heap. `10^6` to `10^7` ints is normal (4 to 40 MB).
