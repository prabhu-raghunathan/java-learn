---
title: "Sorting, Comparable & Comparator"
sidebar_position: 1
---

:::tip[TL;DR]
**`Comparable`** is an object's one built-in "natural" order (`compareTo`, defined inside the class). **`Comparator`** is any number of external orders you pass to `sort`, `PriorityQueue`, `TreeMap`... Write comparators with lambdas or the `Comparator.comparing(...)` builders, chain tie-breakers with `thenComparing`, flip with `reversed()`, and compare numbers with **`Integer.compare`**, not subtraction.
:::

## Complexity at a Glance

| Call | Time | Extra space | Stable |
| --- | --- | --- | --- |
| `Arrays.sort(int[] / long[] / char[] ...)` | O(n log n) | O(log n) typical | n/a |
| `Arrays.sort(T[])`, `Arrays.sort(T[], cmp)`, `List.sort`, `Collections.sort` | O(n log n) comparisons; **O(n)** if already sorted | O(n) | **Yes** |
| `Arrays.sort(arr, from, to)` | O(k log k) for k = to − from | as above | |
| `stream().sorted()` | O(n log n) | O(n) (collects into an array first) | Yes |
| One comparison | O(1) for numbers; **O(L)** for strings of length L; O(cost of your lambda) | | |
| Total with expensive comparisons | O(n log n × comparison cost) | | |
| Building a comparator (`comparing`, `thenComparing`, `reversed`) | O(1) | O(1) | |

The full table of sorting algorithms (bubble, insertion, merge, quick, heap, counting, radix...) is on the [Time & Space Complexity](../../00-complexity-analysis.md#sorting-algorithms) page.

## The Contract

A comparison returns an `int`:

| Return | Meaning |
| --- | --- |
| negative | `a` comes **before** `b` |
| `0` | equal (for `TreeSet`/`TreeMap`: treated as a duplicate) |
| positive | `a` comes **after** `b` |

Rule of thumb: `(a, b) -> compare(a, b)` = ascending, `(a, b) -> compare(b, a)` = descending.

## Comparable: the Natural Order

Implemented **inside** the class. `Collections.sort(list)`, `TreeSet`, `PriorityQueue` use it when no comparator is given. `Integer`, `String`, `Character`, `Long`, `Double` already implement it.

```java title="Emp.java" showLineNumbers
class Emp implements Comparable<Emp> {
    String n; int s;
    Emp(String n, int s) { this.n = n; this.s = s; }

    // highlight-next-line
    @Override public int compareTo(Emp o) { return Integer.compare(s, o.s); }   // by salary
    @Override public String toString() { return n + ":" + s; }
}

List<Emp> es = new ArrayList<>(List.of(new Emp("a", 300), new Emp("b", 100), new Emp("c", 200)));
Collections.sort(es);                       // [b:100, c:200, a:300]
es.sort(Comparator.reverseOrder());         // [a:300, c:200, b:100]
```

## Comparator: Any Order You Like

The same `Person` list sorted four ways, from oldest syntax to newest:

```java showLineNumbers
List<Person> people = new ArrayList<>(List.of(
        new Person("John", 30), new Person("Alice", 25), new Person("Bob", 30)));

// 1. anonymous class (pre-Java 8 style)
Collections.sort(people, new Comparator<Person>() {
    @Override
    public int compare(Person p1, Person p2) {
        return p1.age - p2.age;
    }
});                                          // [Alice (25), John (30), Bob (30)]

// 2. lambda
people.sort((p1, p2) -> p2.age - p1.age);    // [John (30), Bob (30), Alice (25)]  descending

// 3. key extractor
people.sort(Comparator.comparingInt(Person::getAge));          // [Alice (25), John (30), Bob (30)]
people.sort(Comparator.comparing(Person::getName));            // [Alice (25), Bob (30), John (30)]

// 4. chained: age descending, then name ascending
// highlight-next-line
people.sort(Comparator.comparingInt(Person::getAge).reversed().thenComparing(Person::getName));
                                             // [Bob (30), John (30), Alice (25)]
```

### Comparator builder cheat sheet

| Builder | Example | Meaning |
| --- | --- | --- |
| `Comparator.comparing(f)` | `comparing(Person::getName)` | sort by a `Comparable` key |
| `comparingInt / comparingLong / comparingDouble(f)` | `comparingInt(String::length)` | same, no boxing |
| `comparing(f, cmp)` | `comparing(Person::getName, Comparator.reverseOrder())` | key + custom order for that key |
| `.thenComparing(f)` / `.thenComparingInt(f)` | `.thenComparing(Person::getName)` | tie-breaker |
| `.reversed()` | `comparingInt(String::length).reversed()` | flip the **whole chain so far** |
| `Comparator.naturalOrder()` | | ascending natural order |
| `Comparator.reverseOrder()` / `Collections.reverseOrder()` | | descending natural order |
| `Comparator.nullsFirst(cmp)` / `nullsLast(cmp)` | `nullsFirst(naturalOrder())` on `[b, null, a]` | `[null, a, b]` |
| `Map.Entry.comparingByKey()` / `comparingByValue()` | | for sorting map entries |

:::warning[reversed() applies to everything before it]
`comparingInt(Person::getAge).thenComparing(Person::getName).reversed()` reverses **both** keys (age desc, name desc). To reverse only one key, put `.reversed()` right after it, or pass `Comparator.reverseOrder()` as the second argument of `comparing`.
:::

### Lambda type inference with arrays

When the comparator chain starts from a lambda over `int[]`, Java sometimes can't infer the type. Give the first lambda an explicit parameter type:

```java showLineNumbers
Arrays.sort(iv2, Comparator.comparingInt((int[] a) -> a[0]).thenComparingInt(a -> a[1]));
//                                         ^^^^^ needed, otherwise 'a' is an Object
```

Without it, `javac` says: `error: array required, but Object found`.

## Integer.compare vs Subtraction

```java showLineNumbers
Integer[] arr = {Integer.MIN_VALUE, 5, -3};
// error-next-line
Arrays.sort(arr, (a, b) -> a - b);          // ❌ [-3, 5, -2147483648]  overflow broke the order
// success-next-line
Arrays.sort(arr, Integer::compare);         // ✅ [-2147483648, -3, 5]
```

`a - b` overflows when the true difference doesn't fit in an `int`. It's safe for small, non-negative values (indices, lengths, counts) and you'll see it everywhere, but `Integer.compare(a, b)` is never wrong.

| Type | Safe compare |
| --- | --- |
| `int` | `Integer.compare(a, b)` |
| `long` | `Long.compare(a, b)` (never `(int) (a - b)`) |
| `double` | `Double.compare(a, b)` (handles `-0.0` and `NaN`) |
| `char` | `Character.compare(a, b)` or `a - b` (chars are small, can't overflow) |
| `boolean` | `Boolean.compare(a, b)` (`false` < `true`) |
| `String` | `a.compareTo(b)` |

## Common DSA Sorts

### Intervals: start ascending, then end descending

```java showLineNumbers
int[][] iv = {{1, 3}, {2, 2}, {1, 2}};
Arrays.sort(iv, (a, b) -> a[0] != b[0]
        ? Integer.compare(a[0], b[0])           // primary: start ↑
        : Integer.compare(b[1], a[1]));         // tie: end ↓
// [[1, 3], [1, 2], [2, 2]]
```

### Intervals by end (greedy "max non-overlapping")

```java showLineNumbers
int[][] byEnd = {{1, 4}, {2, 3}, {3, 5}};
Arrays.sort(byEnd, (a, b) -> Integer.compare(a[1], b[1]));   // [[2, 3], [1, 4], [3, 5]]
```

### Strings by length, then alphabetically

```java showLineNumbers
String[] w = {"bb", "a", "ccc", "dd"};
Arrays.sort(w, Comparator.comparingInt(String::length).thenComparing(Comparator.naturalOrder()));
// [a, bb, dd, ccc]
```

### Largest number formed by concatenation

The comparator decides by trying both concatenation orders.

```java showLineNumbers
String[] nums = {"3", "30", "34", "5", "9"};
// highlight-next-line
Arrays.sort(nums, (a, b) -> (b + a).compareTo(a + b));
String.join("", nums);                          // "9534330"
```

Each comparison builds two strings of length up to 2L, so this is **O(n log n · L) time** and O(n · L) space.

### Sort indices by their values (keep original positions)

```java showLineNumbers
int[] vals = {30, 10, 20};
Integer[] idx = {0, 1, 2};
Arrays.sort(idx, (i, j) -> Integer.compare(vals[i], vals[j]));
// [1, 2, 0]  → vals[1]=10 is smallest, then vals[2]=20, then vals[0]=30

int[] idx2 = IntStream.range(0, vals.length).boxed()
        .sorted(Comparator.comparingInt(i -> vals[i]))
        .mapToInt(Integer::intValue).toArray();  // [1, 2, 0]  (stream version)
```

**O(n log n) time, O(n) space** for the boxed index array. Sorting `int[][] {value, index}` pairs works too and avoids the lambda's array lookups.

### Records with a natural order

```java showLineNumbers
record P(String name, int age) {}
List<P> ps = new ArrayList<>(List.of(new P("x", 3), new P("y", 1)));
ps.sort(Comparator.comparingInt(P::age));       // [P[name=y, age=1], P[name=x, age=3]]
```

A record can also `implements Comparable<P>` if it has one obvious order (then it works in `TreeSet` / `PriorityQueue` with no comparator).

## After Sorting: Binary Search Bounds

Sorting is often step one; step two is a binary search. `Arrays.binarySearch` returns *any* match, so write these two when duplicates matter. Both are **O(log n) time, O(1) space**.

```java title="Bounds.java" showLineNumbers
// first index with a[i] >= x  (a.length if none)
static int lowerBound(int[] a, int x) {
    int lo = 0, hi = a.length;
    while (lo < hi) {
        int mid = (lo + hi) >>> 1;
        // highlight-next-line
        if (a[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}
// first index with a[i] > x
static int upperBound(int[] a, int x) {
    int lo = 0, hi = a.length;
    while (lo < hi) {
        int mid = (lo + hi) >>> 1;
        // highlight-next-line
        if (a[mid] <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
}
// a = {1, 2, 2, 2, 5, 7}
// lowerBound(a, 2) = 1, upperBound(a, 2) = 4  → count of 2s = 4 - 1 = 3
// lowerBound(a, 6) = 5, lowerBound(a, 9) = 6 (past the end)
```

## Beating O(n log n): Counting Sort

When values are integers in a small range `0..k`, count them instead of comparing: **O(n + k) time, O(k) space**.

```java showLineNumbers
static int[] countingSort(int[] a, int max) {
    int[] c = new int[max + 1];
    for (int v : a) c[v]++;                                 // count each value
    int[] out = new int[a.length];
    int k = 0;
    for (int v = 0; v <= max; v++) while (c[v]-- > 0) out[k++] = v;
    return out;
}
// countingSort({3, 0, 2, 3, 1, 0}, 3) → [0, 0, 1, 2, 3, 3]
```

Typical uses: sort colors (0/1/2), ages, exam scores, letters (`int[26]`), "H-Index".

## Stability

**Stable** = items that compare equal keep their original relative order. Object sorts in Java (`Collections.sort`, `List.sort`, `Arrays.sort(Object[])`) use **TimSort**, which is stable:

```java showLineNumbers
List<P> st = new ArrayList<>(List.of(new P("a", 2), new P("b", 1), new P("c", 2), new P("d", 1)));
st.sort(Comparator.comparingInt(P::age));
// [P[name=b, age=1], P[name=d, age=1], P[name=a, age=2], P[name=c, age=2]]
//         b before d, a before c: original order kept within each age
```

That lets you sort by a secondary key first, then by the primary key, and get a correct two-level sort. With comparators it's simpler to chain `thenComparing`.

| Sort | Algorithm | Time | Extra space | Stable |
| --- | --- | --- | --- | --- |
| `Arrays.sort(int[])` etc. | Dual-Pivot Quicksort | O(n log n) | O(log n) | n/a for primitives |
| `Arrays.sort(T[])`, `List.sort` | TimSort | O(n log n), O(n) if already sorted | O(n) | **Yes** |
| `PriorityQueue` drain | binary heap | O(n log n) | O(n) | No |
| `TreeMap` insert all | red-black tree | O(n log n) | O(n) | n/a (dedupes equal keys) |

## Sorting-Based Patterns

| Pattern | Time | Space | Example |
| --- | --- | --- | --- |
| Sort, then one linear scan | O(n log n) | O(1) to O(n) | Merge Intervals, Meeting Rooms |
| Sort + two pointers | O(n log n) (pairs), O(n²) (triplets) | O(1) | Two Sum II, 3Sum |
| Sort by end time, greedy pick | O(n log n) | O(1) | Non-overlapping Intervals |
| Sort + binary search per query | O((n + q) log n) | O(1) | count elements ≤ x for q queries |
| Custom comparator on strings | O(n log n · L) | O(n · L) | Largest Number, Reorder Log Files |
| Counting / bucket sort | O(n + k) | O(k) | Sort Colors, Top K Frequent, H-Index |

## Gotchas

- A comparator must be **consistent**: if `a < b` and `b < c`, then `a < c`. A broken one (random, or inconsistent tie handling) can make TimSort throw `IllegalArgumentException: Comparison method violates its general contract!`
- `sort(null)` on a `List` means "natural order".
- Sorting a `List<Integer>` descending: `list.sort(Collections.reverseOrder())` → `[3, 2, 1]`.
- `Comparator.comparing(...)` on an `int` getter boxes every key; `comparingInt` avoids that and is faster in tight loops.
- Sorting inside a loop is a classic hidden O(n² log n). Sort once, or keep a `TreeMap` / `PriorityQueue`.
