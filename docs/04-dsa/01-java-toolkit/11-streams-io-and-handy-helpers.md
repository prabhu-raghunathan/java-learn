---
title: "Streams, Fast I/O & Handy Helpers"
sidebar_position: 11
---

:::tip[TL;DR]
Streams turn "loop, filter, collect" into one readable line and shine for grouping and counting (`Collectors.groupingBy`, `counting`, `joining`, `toMap`). They are slower than plain loops, so keep hot inner loops as `for` loops. For competitive-programming style input, use **`BufferedReader` + `StringTokenizer`** and print through a **`PrintWriter`** or `StringBuilder`; `Scanner` + `System.out.println` is often too slow for 10⁵+ lines.
:::

## Streams You'll Actually Use

### Number ranges

```java showLineNumbers
IntStream.range(0, 5);                          // 0 1 2 3 4        (end exclusive)
IntStream.rangeClosed(1, 5).sum();              // 15               (end inclusive)
IntStream.of(3, 1, 2).max().getAsInt();         // 3
IntStream.range(0, 5).boxed().collect(Collectors.toList());   // [0, 1, 2, 3, 4]
IntStream.iterate(0, i -> i < 10, i -> i + 3);  // 0 3 6 9          (Java 9: a for-loop as a stream)
Stream.iterate(1, v -> v * 2).limit(5);         // 1 2 4 8 16
```

### The core operations

```java showLineNumbers
List<Integer> l = List.of(5, 2, 8, 3);
l.stream().filter(v -> v % 2 == 0).toList();    // [2, 8]          keep matching
l.stream().map(v -> v * v).toList();            // [25, 4, 64, 9]  transform
l.stream().mapToInt(Integer::intValue).sum();   // 18              to IntStream for sum/max/average
l.stream().sorted().toList();                   // [2, 3, 5, 8]
l.stream().anyMatch(v -> v > 7);                // true
l.stream().allMatch(v -> v > 0);                // true
l.stream().noneMatch(v -> v < 0);               // true
l.stream().count();                             // 4
l.stream().reduce(0, Integer::sum);             // 18              fold into one value
l.stream().max(Integer::compare).get();         // 8
l.stream().limit(2).toList();                   // [5, 2]
l.stream().skip(2).toList();                    // [8, 3]
Stream.of(1, 2, 2, 3).distinct().toList();      // [1, 2, 3]
```

| Terminal | Returns | Note |
| --- | --- | --- |
| `.toList()` | unmodifiable `List` | Java 16+ |
| `.collect(Collectors.toList())` | mutable `ArrayList` (in practice) | works on every version |
| `.toArray()` on `IntStream` | `int[]` | |
| `.collect(Collectors.toCollection(TreeSet::new))` | any collection you choose | `[2, 5, 8]` for `5, 2, 8` |

### Primitive streams ↔ objects

| From → To | Code |
| --- | --- |
| `int[]` → `IntStream` | `Arrays.stream(arr)` |
| `IntStream` → `Stream<Integer>` | `.boxed()` |
| `Stream<Integer>` → `IntStream` | `.mapToInt(Integer::intValue)` or `.mapToInt(i -> i)` |
| `IntStream` → `Stream<Character>` (from `"abc".chars()`) | `.mapToObj(c -> (char) c)` |
| `int[][]` → all values | `Arrays.stream(grid).flatMapToInt(Arrays::stream)` |
| `List<List<Integer>>` → flat | `nested.stream().flatMap(List::stream).toList()` |

```java showLineNumbers
int[][] grid = {{1, 2}, {3, 4}};
Arrays.stream(grid).flatMapToInt(Arrays::stream).sum();     // 10
IntStream.of(1, 2, 3).summaryStatistics();
// IntSummaryStatistics{count=3, sum=6, min=1, average=2.000000, max=3}
```

## Collectors for Grouping and Counting

```java showLineNumbers
String[] w = {"apple", "avocado", "banana", "blueberry", "cherry"};

Arrays.stream(w).collect(Collectors.groupingBy(s -> s.charAt(0)));
// {a=[apple, avocado], b=[banana, blueberry], c=[cherry]}

Arrays.stream(w).collect(Collectors.groupingBy(s -> s.charAt(0), Collectors.counting()));
// {a=2, b=2, c=1}      (values are Long)

TreeMap<Integer, List<String>> byLen = Arrays.stream(w)
        .collect(Collectors.groupingBy(String::length, TreeMap::new, Collectors.toList()));
// {5=[apple], 6=[banana, cherry], 7=[avocado], 9=[blueberry]}   (sorted keys)

"banana".chars().mapToObj(c -> (char) c)
        .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
// {a=3, b=1, n=2}      character frequencies in one line

Arrays.stream(w).collect(Collectors.partitioningBy(s -> s.length() > 5));
// {false=[apple], true=[avocado, banana, blueberry, cherry]}

Arrays.stream(w).collect(Collectors.joining(", ", "[", "]"));
// [apple, avocado, banana, blueberry, cherry]
Stream.of(1, 2, 3).map(String::valueOf).collect(Collectors.joining());   // "123"
```

:::danger[toMap throws on duplicate keys]
```java showLineNumbers
// error-next-line
Stream.of("a", "a").collect(Collectors.toMap(s -> s, s -> 1));
// ❌ IllegalStateException: Duplicate key a (attempted merging values 1 and 1)

// success-next-line
Stream.of("a", "a", "b").collect(Collectors.toMap(s -> s, s -> 1, Integer::sum));   // ✅ {a=2, b=1}
```
The third argument is the merge function for colliding keys.
:::

:::warning[Streams in interviews]
Streams are great for the setup (counting, grouping, converting). For the algorithm itself (two pointers, DP tables, BFS), write loops: they're faster, easier to debug, and interviewers can follow your index logic. Also, a stream can be consumed only once; a second terminal operation on the same stream throws `IllegalStateException`.
:::

## Functional Interfaces (for lambdas and memo helpers)

From `java.util.function`:

| Interface | Shape | Example |
| --- | --- | --- |
| `Function<T, R>` | `T → R` | `Function<Integer, Integer> sq = v -> v * v;` `sq.apply(4)` → `16` |
| `BiFunction<T, U, R>` | `(T, U) → R` | `BiFunction<Integer, Integer, Integer> add = Integer::sum;` → `5` for `(2, 3)` |
| `Predicate<T>` | `T → boolean` | `Predicate<String> empty = String::isEmpty;` |
| `Supplier<T>` | `() → T` | `Supplier<List<Integer>> sup = ArrayList::new;` |
| `Consumer<T>` | `T → void` | `Consumer<String> pr = System.out::print;` |
| `UnaryOperator<T>` | `T → T` | `UnaryOperator<String> up = String::toUpperCase;` |
| `IntBinaryOperator` | `(int, int) → int` | `(a, b) -> a * b` (no boxing) |

`sq.andThen(v -> v + 1).apply(3)` → `10` (compose two functions).

## Memoisation: Two Ways

```java showLineNumbers
// HashMap memo: works for any key (strings, pairs as records, ...)
static long fib(int n, Map<Integer, Long> memo) {
    if (n < 2) return n;
    Long c = memo.get(n);
    if (c != null) return c;
    long r = fib(n - 1, memo) + fib(n - 2, memo);
    memo.put(n, r);
    return r;
}

// array memo: faster when the state is small integers
static long fib2(int n, long[] dp) {               // dp pre-filled with -1
    if (n < 2) return n;
    if (dp[n] != -1) return dp[n];
    return dp[n] = fib2(n - 1, dp) + fib2(n - 2, dp);
}
// fib(50) = fib2(50) = 12586269025
```

:::danger[computeIfAbsent for recursive memo]
`memo.computeIfAbsent(n, k -> fib(k - 1) + fib(k - 2))` looks neat but **modifies the map from inside its own compute call**. `HashMap` throws `ConcurrentModificationException` for that since Java 9. Use the explicit get/put form above.
:::

## Recursion Depth

The default thread stack fits only a few thousand to a few tens of thousands of frames (about 21,600 trivial frames in a quick test on JDK 21; real methods with more locals fit fewer). Deep DFS on a 10⁵-node path graph can throw `StackOverflowError`.

| Fix | How |
| --- | --- |
| Iterate instead | Use an explicit `ArrayDeque` stack for DFS |
| Bigger stack | Run the solution in a thread with a large stack (works on Codeforces-style judges) |

```java showLineNumbers
Thread t = new Thread(null, () -> solve(), "main", 1 << 26);   // 64 MB stack
t.start();
// the same trivial recursion reached ~1,675,000 frames
```

## Fast Input / Output

### Template: BufferedReader + StringTokenizer + PrintWriter

```java title="Main.java" showLineNumbers
import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));

        int n = Integer.parseInt(br.readLine().trim());
        // highlight-next-line
        StringTokenizer st = new StringTokenizer(br.readLine());   // splits on whitespace
        long sum = 0;
        for (int i = 0; i < n; i++) sum += Integer.parseInt(st.nextToken());

        out.println(sum);
        // highlight-next-line
        out.flush();                                               // forget this and nothing prints
    }
}
// input "5\n1 2 3 4 5" → 15
```

### Reusable FastReader (tokens may span lines)

```java title="FastReader.java" showLineNumbers
static class FastReader {
    private final BufferedReader br;
    private StringTokenizer st;

    FastReader(InputStream in) { br = new BufferedReader(new InputStreamReader(in)); }

    String next() throws IOException {
        while (st == null || !st.hasMoreTokens()) {
            String line = br.readLine();
            if (line == null) return null;                         // end of input
            st = new StringTokenizer(line);
        }
        return st.nextToken();
    }
    int nextInt() throws IOException { return Integer.parseInt(next()); }
    long nextLong() throws IOException { return Long.parseLong(next()); }
}
```

For lots of output lines, append to one `StringBuilder` and print once at the end (`sb.append(x).append('\n')`), or use the `PrintWriter` above.

| Approach | Speed | Notes |
| --- | --- | --- |
| `Scanner` | slow (regex parsing) | fine for small inputs and quick tests |
| `BufferedReader` + `StringTokenizer` | fast | the standard competitive template |
| `DataInputStream` custom reader | fastest | only for extreme limits |
| `System.out.println` per line | slow (flushes) | OK for a few lines |
| `PrintWriter` / one `StringBuilder` | fast | flush / print once |

:::warning[The Scanner nextInt + nextLine trap]
`nextInt()` leaves the newline in the buffer, so the following `nextLine()` returns an **empty string**:

```java showLineNumbers
Scanner sc = new Scanner(System.in);        // input: "3\nhello world\n"
int n = sc.nextInt();
// error-next-line
String line = sc.nextLine();                // ❌ "" (rest of the "3" line)
// success-next-line
sc.nextLine(); String line2 = sc.nextLine(); // ✅ consume the leftover newline first → "hello world"
```
:::

## Small Helpers Worth Knowing

```java showLineNumbers
Objects.equals(a, b);                       // null-safe equals: Objects.equals(null, null) → true
Objects.hash(1, 2);                         // 994  (for writing hashCode)
Objects.requireNonNullElse(x, "def");       // x, or "def" if x is null
Optional<Integer> mx = l.stream().filter(v -> v > 100).max(Integer::compare);
mx.orElse(-1);                              // -1  (no element matched)

var map = new HashMap<String, List<Integer>>();   // var (Java 10): infer the local type
for (var e : map.entrySet()) { }

System.out.printf("%d %s%n", 1, "x");       // formatted print: "1 x"
long t0 = System.nanoTime();                // time a piece of code (difference of two calls)
```

### Pairs and tuples: pick one

| Option | Example | Good for |
| --- | --- | --- |
| `int[]` | `new int[]{r, c}` | queues / heaps (fast, but no `equals`) |
| `record` | `record Pair(int a, int b) {}` → `Pair[a=1, b=2]` | map/set keys, readable code |
| `Map.entry(k, v)` | `Map.entry(1, 2)` → `1=2` | quick immutable pair |
| `long` encoding | `(long) a << 32 \| b` (b ≥ 0) | hashing pairs very fast; a negative `b` sign-extends and wipes out `a` |

Java has no built-in `Pair` class in `java.util` (the `javafx.util.Pair` seen in old answers is not available on most judges).

## Gotchas

- `Stream.toList()` is **unmodifiable**: `add` throws `UnsupportedOperationException`. Use `Collectors.toList()` or wrap in `new ArrayList<>(...)`.
- `groupingBy(..., counting())` gives `Long` values, not `Integer`.
- `IntStream.sum()` returns an `int` and overflows silently; use `asLongStream().sum()` or `mapToLong`.
- `printf` / `String.format` with `%d` on a `double` throws `IllegalFormatConversionException`; use `%f` / `%.2f`.
