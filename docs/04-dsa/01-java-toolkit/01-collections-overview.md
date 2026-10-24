---
title: "Collections Framework & Picking a Structure"
sidebar_position: 1
---

:::tip[TL;DR]
The **Java Collections Framework (JCF)** is Java's equivalent of C++'s STL: ready-made lists, sets, queues, deques, heaps and maps in `java.util`. In DSA, half the battle is picking the right one. Declare variables by **interface** (`List`, `Map`, `Deque`), create them with the **implementation** that has the complexity you need (`ArrayList`, `HashMap`, `ArrayDeque`).
:::

## The Imports You Need

Put these at the top of every DSA file and forget about them.

```java showLineNumbers
import java.util.*;            // collections, Arrays, Collections
import java.util.function.*;   // Function, Predicate (lambdas)
import java.util.stream.*;     // IntStream, Collectors
import java.io.*;              // BufferedReader, PrintWriter
```

## The Family Tree

```mermaid
flowchart TB
    IT["Iterable"] --> C["Collection"]
    C --> L["List<br/>ordered, index access, duplicates OK"]
    C --> S["Set<br/>no duplicates"]
    C --> Q["Queue<br/>FIFO / priority"]
    Q --> D["Deque<br/>both ends"]
    L --> AL["ArrayList"]
    L --> LL["LinkedList"]
    D --> LL
    D --> AD["ArrayDeque"]
    Q --> PQ["PriorityQueue"]
    S --> HS["HashSet"]
    S --> LHS["LinkedHashSet"]
    S --> TS["TreeSet (SortedSet / NavigableSet)"]
    M["Map (not a Collection)"] --> HM["HashMap"]
    M --> LHM["LinkedHashMap"]
    M --> TM["TreeMap (SortedMap / NavigableMap)"]

    style AL fill:#2563eb,color:#fff
    style AD fill:#2563eb,color:#fff
    style PQ fill:#2563eb,color:#fff
    style HS fill:#2563eb,color:#fff
    style HM fill:#2563eb,color:#fff
    style TM fill:#2563eb,color:#fff
    style TS fill:#2563eb,color:#fff
```

The blue boxes are the ones you will use in 95% of problems. `Map` is part of the framework but does **not** extend `Collection`.

## Program to the Interface

```java showLineNumbers
List<Integer> list = new ArrayList<>();          // swap to LinkedList without touching other code
Deque<Integer> stack = new ArrayDeque<>();       // a stack
Queue<Integer> queue = new ArrayDeque<>();       // a FIFO queue
Map<String, List<Integer>> graph = new HashMap<>();
```

- The `<>` (**diamond**) lets the compiler infer the type arguments from the left side.
- Collections hold **objects only**. `List<int>` does not compile; use `List<Integer>`. Java **autoboxes** `int` → `Integer` on the way in and **unboxes** on the way out, which costs time and memory. For hot loops over numbers, a plain `int[]` beats `List<Integer>`.

## Which Structure for Which Job

| I need to… | Use | Key methods | Cost |
| --- | --- | --- | --- |
| Store items, access by index | `ArrayList` | `add`, `get(i)`, `set(i, x)`, `size` | O(1) amortised add, O(1) get |
| Check "have I seen this?" | `HashSet` | `add`, `contains`, `remove` | O(1) average |
| Count / map key → value | `HashMap` | `getOrDefault`, `merge`, `computeIfAbsent` | O(1) average |
| Stack (LIFO) | `ArrayDeque` | `push`, `pop`, `peek` | O(1) |
| Queue (FIFO), BFS | `ArrayDeque` | `offer`, `poll`, `peek` | O(1) |
| Both ends (sliding window) | `ArrayDeque` | `offerFirst/Last`, `pollFirst/Last`, `peekFirst/Last` | O(1) |
| Always get the min / max | `PriorityQueue` | `offer`, `poll`, `peek` | O(log n) / O(log n) / O(1) |
| Sorted unique items + "next bigger / smaller" | `TreeSet` | `floor`, `ceiling`, `lower`, `higher`, `first`, `last` | O(log n) |
| Sorted keys + range queries | `TreeMap` | `floorKey`, `ceilingKey`, `firstKey`, `headMap`, `tailMap` | O(log n) |
| Remember insertion order | `LinkedHashMap` / `LinkedHashSet` | same as Hash versions | O(1) average |
| LRU cache | `LinkedHashMap(cap, 0.75f, true)` | override `removeEldestEntry` | O(1) |

## Big-O Cheat Sheet

Time per operation, and memory. "Bytes/elem" was measured on JDK 21 with 10⁶ `Integer` elements (64-bit JVM, compressed references); see [Time & Space Complexity](../00-complexity-analysis.md) for the method.

| Structure | Add | Remove | Search / contains | Get by index | Ordered? | Space | Bytes/elem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `int[]` (for comparison) | n/a (fixed size) | n/a | O(n), O(log n) if sorted | **O(1)** | as written | O(n) | 4 |
| `ArrayList` | O(1)* at end, O(n) in middle | O(n) (shift), O(1) at end | O(n) | **O(1)** | insertion | O(n) | ~21 |
| `LinkedList` | O(1) at ends | O(1) at ends, O(n) by value | O(n) | O(n) | insertion | O(n) | 40 |
| `ArrayDeque` | O(1)* at ends | O(1) at ends | O(n) | n/a | insertion | O(n) | ~21 |
| `PriorityQueue` | O(log n) | O(log n) `poll`, **O(n)** `remove(x)` | O(n) | n/a | **heap order only** | O(n) | ~21 |
| `HashSet` | O(1) avg | O(1) avg | **O(1) avg** | n/a | none | O(n) | ~56 |
| `HashMap` | O(1) avg | O(1) avg | O(1) avg (key) | n/a | none | O(n) | ~72 |
| `LinkedHashSet` / `LinkedHashMap` | O(1) avg | O(1) avg | O(1) avg | n/a | insertion (or access) | O(n) | ~80 (map) |
| `TreeSet` / `TreeMap` | O(log n) | O(log n) | **O(log n)** | n/a | **sorted** | O(n) | ~56 / ~72 |

\* amortised: the backing array occasionally grows (ArrayList by 1.5×; ArrayDeque doubles while small, then grows by 1.5×), which is O(n) once but averages out to O(1).

Every structure here is **O(n) space**, but the constant differs a lot: a `HashMap<Integer, Integer>` uses ~18× the memory of an `int[]` holding the same numbers, because each entry is a node object plus two boxed `Integer`s.

### Searching: HashSet vs TreeSet vs List vs sorted array

| Structure | `contains` time | Extra space | Also gives you | Pick it when |
| --- | --- | --- | --- | --- |
| `HashSet` | **O(1)** average, O(log n) worst | O(n) | nothing else | you only ask "is x present?" |
| `TreeSet` | O(log n) | O(n) | sorted order, `floor`/`ceiling`, min/max | you also need order or the nearest value |
| sorted `int[]` + `Arrays.binarySearch` | O(log n) | O(1) beyond the array | insertion point | the data is fixed after one sort (O(n log n) once) |
| `ArrayList.contains` | **O(n)** | O(1) beyond the list | index access | never inside a loop |

Measured (10⁶ lookups in 10⁶ `Integer`s): `HashSet` 72 ms, sorted `int[]` 165 ms, `TreeSet` 952 ms; `ArrayList.contains` took 1.1 s for just 1,000 lookups.

:::info[Hash worst case]
Since Java 8, a `HashMap` bucket that collects 8 or more colliding keys (in a table of at least 64 buckets) turns into a **red-black tree**, so the worst case per operation is **O(log n)**, not O(n). In interviews, still quote **O(1) average**.
:::

## Legacy Classes: Know Them, Don't Use Them

| Legacy | Why avoid | Use instead |
| --- | --- | --- |
| `Vector` | Every method is `synchronized`, so it is slower for single-threaded code | `ArrayList` |
| `Stack` | Extends `Vector` (synchronized), and iterates bottom → top, which surprises people | `ArrayDeque` (`push`/`pop`/`peek`) |
| `Hashtable` | Synchronized, no `null` keys or values | `HashMap` (single thread), `ConcurrentHashMap` (multi-thread) |

`Stack` still works and plenty of solutions use it, so you should be able to read it. The javadoc itself recommends `Deque` for new code.

## Storing Your Own Classes

Collections can hold any object, including your own classes.

```java title="Person.java" showLineNumbers
class Person {
    String name;
    int age;

    Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {                 // controls how println shows it
        return name + " (" + age + ")";
    }
}
```

```java showLineNumbers
List<Person> people = new ArrayList<>();
people.add(new Person("John", 30));
people.add(new Person("Alice", 25));
System.out.println(people);                    // [John (30), Alice (25)]
```

Without `toString()` you would see something like `Person@1b6d3586`.

### Hash-based collections need `equals` + `hashCode`

`HashSet`, `HashMap` keys and `contains` use **`hashCode()` to find the bucket and `equals()` to compare**. The defaults from `Object` compare **identity** (same object in memory), so two objects with the same fields count as different.

```java title="❌ No equals/hashCode: duplicates sneak in" showLineNumbers
class BadPoint { int x, y; BadPoint(int x, int y) { this.x = x; this.y = y; } }

Set<BadPoint> set = new HashSet<>();
set.add(new BadPoint(1, 2));
set.add(new BadPoint(1, 2));
// error-next-line
System.out.println(set.size());                        // ❌ 2
// error-next-line
System.out.println(set.contains(new BadPoint(1, 2)));  // ❌ false
```

```java title="✅ Override both" showLineNumbers
class Point {
    int x, y;
    Point(int x, int y) { this.x = x; this.y = y; }

    // success-start
    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Point p)) return false;
        return x == p.x && y == p.y;
    }
    @Override public int hashCode() { return Objects.hash(x, y); }
    // success-end
}
// set.size() -> 1, set.contains(new Point(1, 2)) -> true
```

### The shortcut: `record` <Since v="16" />

A **record** generates the constructor, getters, `equals`, `hashCode` and `toString` for you. It is the cleanest way to make a pair / tuple / grid cell key in DSA.

```java showLineNumbers
record Cell(int r, int c) {}

Set<Cell> seen = new HashSet<>();
seen.add(new Cell(0, 0));
// highlight-next-line
System.out.println(seen.contains(new Cell(0, 0)));     // true
System.out.println(new Cell(2, 3));                    // Cell[r=2, c=3]

Map<Cell, Integer> dist = new HashMap<>();
dist.put(new Cell(1, 1), 5);
System.out.println(dist.get(new Cell(1, 1)));          // 5
System.out.println(new Cell(2, 3).r());                // accessor is r(), not getR()
```

A record can be declared inside a method or as a nested class in your `Solution`, which is handy on LeetCode.

### Other ways to use a pair as a key

| Key type | Works as a HashSet / HashMap key? | Cost per hash / equals | Extra memory per key | Notes |
| --- | --- | --- | --- | --- |
| `record Cell(int r, int c)` | ✅ yes | O(1) | one small object | Best choice on Java 16+ |
| `List.of(r, c)` / `Arrays.asList(r, c)` | ✅ yes | O(k) for k items, recomputed every time | list + boxed `Integer`s | slower; fine for small tuples |
| `r + "," + c` (String) | ✅ yes | O(digits) + building the string each lookup | a new `String` per lookup | easy but slow |
| `r * cols + c` (int) | ✅ yes | O(1) | none (`int`, boxed only inside the map) | Fastest; decode with `id / cols`, `id % cols` |
| `(long) r << 32 \| c` (long) | ✅ yes | O(1) | none | Fast, works for large coordinates (needs `c ≥ 0`) |
| `int[] {r, c}` | ❌ **no** | identity only | | Arrays use identity `equals`: `contains(new int[]{1, 2})` is `false` |

```java showLineNumbers
int r = 3, c = 4, cols = 10;
int id = r * cols + c;                                 // 34
System.out.println(id / cols + " " + id % cols);       // 3 4   (back to r and c)
```

:::danger[Never mutate a key after inserting it]
If a field used by `hashCode` changes after the object is in a `HashSet` / `HashMap`, the object sits in the wrong bucket and can no longer be found, not even with itself:

```java showLineNumbers
Point p = new Point(1, 1);
Set<Point> s = new HashSet<>();
s.add(p);
p.x = 9;                                     // mutate the key
// error-next-line
System.out.println(s.contains(p));           // ❌ false, though p is in the set
System.out.println(s.size());                // 1
```

Records are immutable, which is another reason to prefer them as keys.
:::

## Hidden Costs of Keys and Elements

- **Hashing a key costs O(key size).** For an `Integer` key it's O(1); for a `String` of length L it's O(L) the first time (then cached in the String); for a `List<Integer>` key it's O(list length) **every time** (lists don't cache their hash). So a "O(1)" map lookup with a 1,000-char string key is really O(1,000).
- **Comparing keys costs O(key size) too.** `TreeMap<String, …>` operations are O(L · log n) for strings of length L.
- **Boxing:** `List<Integer>` stores pointers to `Integer` objects. Values -128..127 come from a shared cache; everything else allocates a 16-byte object per value.
- **`equals`/`hashCode` you write** must be O(fields), never loop over big collections.

## Common Interview Traps

:::warning[ArrayList vs LinkedList]
"Use `LinkedList` for fast insertion" is only true **if you already hold an iterator at that position**. Reaching the position is O(n), and `LinkedList` has poor cache locality, so `ArrayList` wins almost every real benchmark. For a queue or deque, use `ArrayDeque` over both.
:::

- `HashMap` / `HashSet` iteration order is **not** insertion order and not sorted. Never rely on it in output. If the problem wants sorted output, use `TreeMap`, or sort at the end.
- `HashMap` allows one `null` key and `null` values. `TreeMap` throws `NullPointerException` on a `null` key (it can't compare it). `ArrayDeque` and `PriorityQueue` reject `null` elements.
- Interface methods like `List.of`, `Set.of`, `Map.of` return **immutable** collections: `add` throws `UnsupportedOperationException`. Wrap them (`new ArrayList<>(List.of(1, 2))`) if you need to modify.
