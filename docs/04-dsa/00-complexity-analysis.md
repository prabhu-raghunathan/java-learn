---
title: "Time & Space Complexity (Big-O Cheat Sheet)"
sidebar_position: 0
---

:::tip[TL;DR]
**Time complexity** is how the number of steps grows with the input size `n`; **space complexity** is how much **extra** memory grows with `n`. Interviews expect both for every solution, stated as Big-O of the **worst case** (or average, if you say so), and a one-line justification ("one pass over the array, plus a hash set of up to n elements: O(n) time, O(n) space").
:::

## Big-O, Big-Ω, Big-Θ

| Notation | Meaning | Everyday reading |
| --- | --- | --- |
| **O(f(n))** | upper bound: grows **no faster** than f(n) | "at most this slow" (what interviews mean by "complexity") |
| **Ω(f(n))** | lower bound: grows **at least** as fast | "at least this much work" |
| **Θ(f(n))** | tight bound: both O and Ω | "exactly this growth rate" |

Rules for simplifying:

- **Drop constants:** O(2n + 10) → O(n). O(n/2) → O(n).
- **Keep the dominant term:** O(n² + n log n + n) → O(n²).
- **Different inputs keep different letters:** looping over `a` then over `b` is O(a + b); nested is O(a · b). Don't call it O(n²) unless both are `n`.
- **log base doesn't matter:** log₂ n and log₁₀ n differ by a constant.

## The Growth Ladder

| Complexity | Name | n = 10 | n = 1,000 | n = 10⁶ | Typical source |
| --- | --- | --- | --- | --- | --- |
| O(1) | constant | 1 | 1 | 1 | array index, `HashMap.get`, `push`/`pop` |
| O(log n) | logarithmic | 3 | 10 | 20 | binary search, `TreeMap` ops, heap `offer`/`poll` |
| O(√n) | square root | 3 | 32 | 1,000 | primality by trial division |
| O(n) | linear | 10 | 1,000 | 10⁶ | one pass, two pointers, sliding window |
| O(n log n) | linearithmic | 33 | 10⁴ | 2 × 10⁷ | sorting, n heap ops, divide & conquer |
| O(n²) | quadratic | 100 | 10⁶ | 10¹² ❌ | nested loops over pairs, DP on two strings |
| O(n³) | cubic | 1,000 | 10⁹ ❌ | | triple loops, Floyd–Warshall |
| O(2ⁿ) | exponential | 1,024 | ❌ | | all subsets, naive recursion |
| O(n!) | factorial | 3.6 × 10⁶ | ❌ | | all permutations |

## Input Size → Complexity You Can Afford

A rule of thumb used by competitive programmers: a judge runs roughly **10⁸ simple operations per second** (Java is a bit slower than C++ to start up, so leave headroom). Read the constraints first; they tell you the intended complexity.

| Constraint on n | Target complexity | Typical techniques |
| --- | --- | --- |
| n ≤ 10–12 | O(n!), O(n · n!) | permutations, brute force |
| n ≤ 20–25 | O(2ⁿ), O(2ⁿ · n) | subsets, bitmask DP, backtracking |
| n ≤ 100–500 | O(n³) | 3 nested loops, Floyd–Warshall, interval DP |
| n ≤ 2,000–5,000 | O(n²) | 2D DP, all pairs |
| n ≤ 10⁵–10⁶ | **O(n log n)** or O(n) | sorting, heaps, `TreeMap`, binary search, two pointers |
| n ≤ 10⁸ | O(n) with a tiny constant, or O(log n) per query | single pass over primitives, math |
| n ≥ 10⁹ (just a number) | O(log n), O(√n) or O(1) | binary search on the answer, math formulas, fast power |

## How to Work Out Time Complexity

| Code shape | Complexity | Why |
| --- | --- | --- |
| `for (i = 0; i < n; i++)` | O(n) | n iterations |
| two loops one after another | O(n + m) | add sequential parts |
| loop inside a loop (both to n) | O(n²) | multiply nested parts |
| `for (j = i; j < n; j++)` inside `for i` | O(n²) | n + (n−1) + … + 1 = n(n+1)/2 |
| `for (i = 1; i < n; i *= 2)` | O(log n) | i doubles each time |
| `while (n > 0) n /= 2;` | O(log n) | n halves each time |
| `for (i = 1; i * i <= n; i++)` | O(√n) | stops at √n |
| outer loop n, inner loop log n | O(n log n) | e.g. n `TreeMap.put` calls |
| two pointers moving only forward | O(n) | each pointer moves at most n times in total |
| monotonic stack: each element pushed and popped once | O(n) | **amortised**: count total pushes/pops, not per-iteration worst case |
| BFS / DFS on a graph | O(V + E) | every vertex and edge handled once |
| BFS / DFS on an R × C grid | O(R · C) | every cell visited once |

```java showLineNumbers
for (int i = 0; i < n; i++)                 // n times
    for (int j = i + 1; j < n; j++)         // n-1, n-2, ... 1 times
        if (a[i] + a[j] == target) ...      // → O(n²) time, O(1) space

for (int i = 1; i < n; i *= 2)              // log n times
    for (int j = 0; j < n; j++) ...         // n times each → O(n log n)
```

:::danger[Hidden loops inside library calls]
A single line can hide O(n) work. These turn an "O(n)" loop into O(n²):

| Inside a loop | Real cost per call | Fix |
| --- | --- | --- |
| `list.contains(x)`, `list.indexOf(x)` | O(n) | use a `HashSet` (O(1)) |
| `list.remove(0)` / `list.add(0, x)` on `ArrayList` | O(n) (shifts everything) | `ArrayDeque.pollFirst` / `offerFirst` (O(1)) |
| `s += c` on a `String` | O(length) (copies the whole string) | `StringBuilder.append` (amortised O(1)) |
| `s.substring(i, j)` | O(j − i) (copies) | pass indices instead of substrings |
| `Collections.max(list)` / `stream().max()` | O(n) | keep a running max, or a heap / `TreeMap` |
| `map.containsValue(v)` | O(n) | keep a reverse map |
| `pq.remove(x)` on a `PriorityQueue` | O(n) | lazy deletion (skip stale entries on `poll`) |
| `Arrays.sort(...)` | O(n log n) | sort once, outside the loop |
:::

A measured example on JDK 21 (one run on a 2-core cloud machine, so only the ratios matter):

| Task (n = 200,000 or 100,000) | Slow version | Fast version |
| --- | --- | --- |
| Remove 200,000 items from the front | `ArrayList.remove(0)`: **1,911 ms** | `ArrayDeque.pollFirst()`: **8 ms** |
| Build a 100,000-char string | `s += "a"`: **1,057 ms** | `StringBuilder.append`: **5 ms** |

## Recursion: Time = Calls × Work per Call

Draw the **recursion tree**: count the nodes (calls) and the work done in each.

| Recurrence | Example | Complexity |
| --- | --- | --- |
| T(n) = T(n−1) + O(1) | factorial, linked-list recursion | O(n) |
| T(n) = T(n/2) + O(1) | binary search | O(log n) |
| T(n) = 2T(n/2) + O(n) | merge sort | O(n log n) |
| T(n) = 2T(n/2) + O(1) | tree traversal, max of an array by halves | O(n) |
| T(n) = T(n−1) + O(n) | selection sort written recursively | O(n²) |
| T(n) = 2T(n−1) + O(1) | naive Fibonacci, Tower of Hanoi | O(2ⁿ) |
| memoised recursion | Fibonacci with memo, top-down DP | O(#states × work per state) |

:::info[Master theorem in one line]
For T(n) = a·T(n/b) + O(nᵈ): if **d > log_b a** → O(nᵈ); if **d = log_b a** → O(nᵈ log n); if **d < log_b a** → O(n^(log_b a)). Merge sort: a = 2, b = 2, d = 1 → equal → O(n log n).
:::

## Space Complexity

| Term | Counts | Example |
| --- | --- | --- |
| **Auxiliary space** | only the extra memory your algorithm allocates | two pointers: O(1) |
| **Total space** | input + auxiliary | an algorithm that reads an `int[n]`: at least O(n) |

Interviewers almost always mean **auxiliary space**. Say which one you mean if it's ambiguous.

What counts as extra space:

| Source | Space |
| --- | --- |
| a few variables, pointers, counters | O(1) |
| a `HashSet` / `HashMap` of up to n items | O(n) |
| a copy of the input, a new array, `toCharArray()`, `substring` | O(n) (or the copied length) |
| a fixed-size count array like `int[26]` / `int[128]` | **O(1)** (doesn't grow with n; some people say O(alphabet)) |
| a DP table of R × C | O(R · C) (often reducible to O(C) by keeping one row) |
| **recursion** depth d | **O(d)** call stack, even though you allocated nothing |
| BFS queue | O(width of the graph), up to O(V) |
| DFS (recursive or explicit stack) | O(depth), up to O(V) |
| the output itself (e.g. all subsets) | usually **not** counted, but mention it: O(2ⁿ · n) to store all subsets |

:::warning[Recursion is not free]
Recursive DFS on a 10⁵-node chain uses O(10⁵) stack frames and can throw `StackOverflowError` in Java. "Recursive binary search" is O(log n) space; the iterative one is O(1). Always count the call stack.
:::

### What things really cost in memory (Java)

Big-O hides constants, but in Java the constant is big when you box numbers. Measured on JDK 21 (64-bit, compressed references), 10⁶ elements:

| Structure | Bytes per element | vs `int[]` |
| --- | --- | --- |
| `int[]` | 4 | 1× |
| `long[]` | 8 | 2× |
| `Integer[]` / `ArrayList<Integer>` / `ArrayDeque<Integer>` / `PriorityQueue<Integer>` | ~20–21 | ~5× |
| `LinkedList<Integer>` | 40 | 10× |
| `HashSet<Integer>` / `TreeSet<Integer>` | ~56 | 14× |
| `HashMap<Integer, Integer>` / `TreeMap<Integer, Integer>` | ~72 | 18× |
| `LinkedHashMap<Integer, Integer>` | ~80 | 20× |

All of these are O(n) space, but a `HashMap` of 10⁷ entries needs ~720 MB while an `int[]` of 10⁷ needs 40 MB. When keys are small integers (0..n), an array beats a map.

## Amortised Complexity

**Amortised O(1)** = an operation is occasionally expensive, but the average over any sequence of operations is O(1).

- `ArrayList.add`: when the backing array is full it copies everything into a 1.5× bigger array (O(n) once), but over n adds the total copying is O(n), so each add is amortised O(1).
- `StringBuilder.append`, `ArrayDeque.offer`, `HashMap.put` (resizing at 75% full) work the same way.
- Monotonic stack / sliding window: one iteration may pop many elements, but each element is popped at most once overall, so the whole loop is O(n).

## Java Collections: Master Table

| Structure | Access | Search / contains | Insert | Delete | Min / max | Space |
| --- | --- | --- | --- | --- | --- | --- |
| `int[]` | O(1) | O(n) (O(log n) if sorted) | n/a (fixed) | n/a | O(n) | O(n) |
| `ArrayList` | O(1) | O(n) | O(1)* end, O(n) middle | O(1) end, O(n) middle | O(n) | O(n) |
| `LinkedList` | O(n) | O(n) | O(1) ends | O(1) ends | O(n) | O(n) |
| `ArrayDeque` (stack/queue) | ends only O(1) | O(n) | O(1)* ends | O(1) ends | O(n) | O(n) |
| `PriorityQueue` | peek O(1) | O(n) | O(log n) | poll O(log n), remove(x) O(n) | **O(1)** one end | O(n) |
| `HashSet` / `HashMap` | n/a / get O(1) | **O(1)** avg, O(log n) worst | O(1) avg | O(1) avg | O(n) | O(n) |
| `LinkedHashMap` | get O(1) | O(1) avg | O(1) avg | O(1) avg | O(n) | O(n) |
| `TreeSet` / `TreeMap` | get O(log n) | **O(log n)** | O(log n) | O(log n) | **O(log n)** both ends | O(n) |
| `String` | `charAt` O(1) | `indexOf` O(n·m) worst | new string O(n) | new string O(n) | n/a | O(n) |
| `StringBuilder` | O(1) | O(n·m) | O(1)* end, O(n) middle | O(1) end, O(n) middle | n/a | O(n) |

\* amortised.

### Search: HashSet vs TreeSet vs the rest

| Structure | `contains` time | Extra space | Keeps order? | Measured: 10⁶ lookups in 10⁶ items |
| --- | --- | --- | --- | --- |
| `HashSet<Integer>` | **O(1)** average | O(n), ~56 B/elem | no | **72 ms** |
| sorted `int[]` + `Arrays.binarySearch` | O(log n) | O(1) extra (O(n) for the array) | sorted, static | 165 ms |
| `TreeSet<Integer>` | O(log n) | O(n), ~56 B/elem | **sorted, dynamic** | 952 ms |
| `ArrayList<Integer>.contains` | **O(n)** | O(n) | insertion | 1,000 lookups took 1.1 s → 10⁶ would take ~18 minutes |

**Takeaway:** `HashSet` for pure membership; `TreeSet` only when you also need order or "nearest" queries; a sorted array when the data doesn't change.

## Sorting Algorithms

| Algorithm | Best | Average | Worst | Extra space | Stable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Bubble sort | O(n) | O(n²) | O(n²) | O(1) | Yes | best case needs an early-exit flag |
| Selection sort | O(n²) | O(n²) | O(n²) | O(1) | No | fewest swaps (n − 1) |
| Insertion sort | O(n) | O(n²) | O(n²) | O(1) | Yes | great for small or nearly sorted input |
| Merge sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | linked lists: O(1) extra (O(log n) stack if recursive) |
| Quick sort | O(n log n) | O(n log n) | O(n²) | O(log n) stack | No | bad pivots → O(n²); randomise the pivot |
| Heap sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No | in place, but poor cache use |
| Counting sort | O(n + k) | O(n + k) | O(n + k) | O(k) | Yes | values in a small range 0..k |
| Radix sort | O(d · (n + b)) | O(d · (n + b)) | O(d · (n + b)) | O(n + b) | Yes | d digits in base b |
| Bucket sort | O(n + k) | O(n + k) | O(n²) | O(n + k) | Yes | uniformly distributed values |
| **TimSort** (Java objects) | **O(n)** | O(n log n) | O(n log n) | O(n) | **Yes** | `Collections.sort`, `List.sort`, `Arrays.sort(Object[])` |
| **Dual-Pivot Quicksort** (Java primitives) | O(n log n) | O(n log n) | O(n log n) (heapsort fallback since JDK 14) | O(log n), may use an O(n) buffer for partly sorted runs | No | `Arrays.sort(int[])` |

Comparison sorts can't beat **Ω(n log n)** in the worst case; counting/radix/bucket sorts beat it by not comparing.

## Common Patterns and Their Complexity

| Pattern | Time | Space | Example problem |
| --- | --- | --- | --- |
| Hash set / map lookup | O(n) | O(n) | Two Sum, Contains Duplicate |
| Sort first, then scan | O(n log n) | O(1) to O(n) | Merge Intervals, 3Sum (O(n²) total) |
| Two pointers | O(n) | O(1) | Valid Palindrome, Container With Most Water |
| Sliding window | O(n) | O(1) or O(k) / O(alphabet) | Longest Substring Without Repeating Characters |
| Prefix sums | O(n) build, O(1) per query | O(n) | Range Sum Query, Subarray Sum Equals K (+ map) |
| Binary search on a sorted array | O(log n) | O(1) | Search Insert Position |
| Binary search on the answer | O(n · log(range)) | O(1) | Koko Eating Bananas |
| Monotonic stack | O(n) | O(n) | Daily Temperatures, Largest Rectangle |
| Monotonic deque | O(n) | O(k) | Sliding Window Maximum |
| Heap of size k | O(n log k) | O(k) | Kth Largest, Top K Frequent |
| BFS / DFS (graph) | O(V + E) | O(V) | Number of Islands, Course Schedule |
| Dijkstra with a heap | O((V + E) log V) | O(V + E) | Network Delay Time |
| Union-Find (path compression + rank) | ~O(α(n)) per op (practically O(1)) | O(n) | Number of Provinces, Redundant Connection |
| Topological sort (Kahn) | O(V + E) | O(V) | Course Schedule II |
| Backtracking: subsets | O(2ⁿ · n) | O(n) stack (+ output) | Subsets |
| Backtracking: permutations | O(n! · n) | O(n) stack (+ output) | Permutations |
| 1D DP | O(n) or O(n · k) | O(n) → often O(1) | Climbing Stairs, House Robber |
| 2D DP (two strings) | O(n · m) | O(n · m) → O(min(n, m)) | LCS, Edit Distance |
| Trie insert / search | O(L) per word | O(total characters) | Implement Trie, Word Search II |

## Saying It in an Interview

1. State **time** and **space**, both in Big-O, and name the variables (`n` = array length, `m` = string length, `V`/`E` for graphs).
2. Justify in one sentence: "Sorting is O(n log n), the scan is O(n), so O(n log n) overall; the hash map holds at most n entries, so O(n) space."
3. Mention the **brute force** first and its cost, then the improvement: "Brute force checks all pairs, O(n²); with a hash map it's O(n) time at the cost of O(n) space." That **time-space trade-off** sentence is what interviewers listen for.
4. Point out hidden costs: hashing a `String` key is O(length), recursion uses stack space, and sorting the input mutates it (copy it if you mustn't).
