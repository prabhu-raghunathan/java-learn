---
title: "Lists, Iterators & the Collections Class"
sidebar_position: 4
---

:::tip[TL;DR]
`ArrayList` is the default dynamic array: O(1) `get`, amortised O(1) `add` at the end, O(n) insert/remove in the middle. `LinkedList` is a doubly linked list that is rarely the right answer. The `Collections` class adds ready-made algorithms: `sort`, `reverse`, `max`, `min`, `frequency`, `binarySearch`, `swap`, `nCopies` and more.
:::

:::tip[Run it yourself]
All snippets run in `jshell`.
:::

## ArrayList

A **resizable array**. When it runs out of room it allocates a new array about 1.5× bigger and copies everything over.

```java showLineNumbers
List<Integer> l = new ArrayList<>(List.of(10, 20, 30));   // mutable copy of an immutable list
l.add(1, 15);                   // insert at index 1, shifts the rest: [10, 15, 20, 30]
l.get(2);                       // 20
l.set(0, 5);                    // returns the OLD value (10); list: [5, 15, 20, 30]
l.size();                       // 4
l.contains(20);                 // true    O(n) linear scan
l.indexOf(30);                  // 3       (-1 if absent)
l.isEmpty();                    // false
l.addAll(List.of(40, 50));      // append another collection
l.clear();                      // remove everything
```

| Operation | Cost | Why |
| --- | --- | --- |
| `get(i)`, `set(i, x)` | O(1) | direct array index |
| `add(x)` at the end | O(1) amortised | occasional resize copies everything |
| `add(i, x)`, `remove(i)` | O(n) | elements after `i` shift |
| `remove(size() - 1)` | O(1) | nothing to shift: use it as a stack |
| `contains`, `indexOf`, `remove(Object)` | O(n) | linear scan |

:::tip[Pre-size when you know n]
`new ArrayList<>(n)` sets the initial **capacity** (room), not the size: the list is still empty and `get(0)` throws. It just avoids resize copies.
:::

### The remove(int) vs remove(Object) trap

`remove` is overloaded. With a `List<Integer>`, an `int` argument means **index**.

```java showLineNumbers
List<Integer> r = new ArrayList<>(List.of(1, 2, 3));
// error-next-line
r.remove(1);                    // ❌ removes INDEX 1 → [1, 3]

List<Integer> r2 = new ArrayList<>(List.of(1, 2, 3));
// success-next-line
r2.remove(Integer.valueOf(1));  // ✅ removes the VALUE 1 → [2, 3]
r2.remove((Integer) 2);         // same thing with a cast
```

### subList is a view

```java showLineNumbers
List<Integer> ab = new ArrayList<>(List.of(1, 2, 3, 4));
ab.subList(1, 3);               // [2, 3]   [from, to)

List<Integer> sl = new ArrayList<>(List.of(1, 2, 3, 4, 5));
sl.subList(1, 3).clear();       // removes a range from the ORIGINAL: [1, 4, 5]
```

Changes through the view affect the original, and structurally modifying the original afterwards makes the view throw `ConcurrentModificationException`. Copy it (`new ArrayList<>(l.subList(a, b))`) if you want to keep it.

### Lists of lists (adjacency lists, results)

```java showLineNumbers
List<List<Integer>> graph = new ArrayList<>();
for (int i = 0; i < n; i++) graph.add(new ArrayList<>());   // one inner list per node
graph.get(u).add(v);                                         // edge u → v
```

:::danger[Add a COPY of the path in backtracking]
```java showLineNumbers
List<List<Integer>> res = new ArrayList<>();
List<Integer> path = new ArrayList<>(List.of(1, 2));
// error-next-line
res.add(path);                  // ❌ stores a reference; later path changes leak in
path.add(3);
System.out.println(res);        // [[1, 2, 3]]

// success-next-line
res.add(new ArrayList<>(path)); // ✅ snapshot of the current path
```
This is the single most common bug in subsets / permutations / combination-sum solutions: every result ends up as the same (often empty) list.
:::

## Immutable Lists

| Factory | Can `set`? | Can `add`/`remove`? | Allows `null`? |
| --- | --- | --- | --- |
| `List.of(1, 2)` (Java 9) | No | No | **No** (NPE) |
| `Arrays.asList(1, 2)` | Yes | No | Yes |
| `Collections.unmodifiableList(l)` | No | No | Yes (a read-only view) |
| `stream.toList()` (Java 16) | No | No | Yes |
| `new ArrayList<>(anyOfTheAbove)` | Yes | Yes | Yes |

Mutating any of the read-only ones throws `UnsupportedOperationException`.

## LinkedList

A **doubly linked list**. It implements both `List` and `Deque`, so it has index methods and end methods.

```java showLineNumbers
LinkedList<String> ll = new LinkedList<>();
ll.add("Cat");
ll.add("Dog");
ll.addFirst("Ant");             // [Ant, Cat, Dog]
ll.getFirst();                  // "Ant"
ll.getLast();                   // "Dog"
ll.removeFirst();               // "Ant" → [Cat, Dog]
```

| | `ArrayList` | `LinkedList` |
| --- | --- | --- |
| `get(i)` | **O(1)** | O(n) (walks from the nearer end) |
| Add/remove at the ends | O(1) at the end only | **O(1) at both ends** |
| Insert/remove in the middle | O(n) shift | O(n) to find + O(1) to relink |
| Memory per element | ~4 bytes (a reference) | ~24 bytes (node with prev/next) |
| Cache friendliness | Contiguous, fast | Scattered, slow |
| Allows `null` | Yes | Yes |

:::warning[When is LinkedList actually right?]
Almost never in DSA. Use `ArrayList` for indexed lists and `ArrayDeque` for queues/stacks. `LinkedList` is only needed if you must store `null` in a queue, or you remove from the middle through an `Iterator` you already hold. For LRU caches, use `LinkedHashMap` or your own node class.
:::

## Iterating and Modifying Safely

```java showLineNumbers
for (int i = 0; i < list.size(); i++) list.get(i);       // index loop (fine for ArrayList)
for (int x : list) { }                                   // for-each (uses an Iterator)
list.forEach(x -> System.out.print(x + " "));            // lambda
for (int i = list.size() - 1; i >= 0; i--) list.get(i); // backwards
```

:::danger[ConcurrentModificationException]
Removing from a list inside a for-each loop breaks the hidden iterator:

```java showLineNumbers
List<Integer> cm = new ArrayList<>(List.of(1, 2, 3, 4));
// error-next-line
for (Integer v : cm) if (v % 2 == 0) cm.remove(v);      // ❌ ConcurrentModificationException
```

```java showLineNumbers
// success-start
cm.removeIf(v -> v % 2 == 0);                             // ✅ [1, 3]  (cleanest)

Iterator<Integer> it = cm.iterator();                     // ✅ or remove through the iterator
while (it.hasNext()) if (it.next() % 2 == 0) it.remove();
// success-end
```
:::

```java showLineNumbers
List<Integer> rp = new ArrayList<>(List.of(1, 2, 3));
rp.replaceAll(v -> v * 10);     // [10, 20, 30]  (transform in place)
```

### ListIterator: both directions + edit in place

`ListIterator` can go forward and backward and can `set` / `add` while iterating.

```java showLineNumbers
List<String> list = new ArrayList<>(List.of("One", "Two"));
ListIterator<String> it = list.listIterator();
while (it.hasNext()) System.out.println(it.next());       // One, Two
while (it.hasPrevious()) System.out.println(it.previous()); // Two, One (now walking back)
```

| Method | Does |
| --- | --- |
| `hasNext()` / `next()` | move forward |
| `hasPrevious()` / `previous()` | move backward |
| `nextIndex()` / `previousIndex()` | index of the element `next()` / `previous()` would return |
| `set(x)` | replace the element last returned |
| `add(x)` | insert before the cursor |
| `remove()` | remove the element last returned |

## Sequenced methods <Since v="21" />

Java 21 added `getFirst`, `getLast`, `addFirst`, `addLast`, `removeFirst`, `removeLast` and `reversed` to every `List` (and to `Deque`, `LinkedHashSet`, `LinkedHashMap`, `TreeSet`, `TreeMap` as `SequencedCollection` / `SequencedMap`).

```java showLineNumbers
List<Integer> bw = List.of(1, 2, 3);
bw.getFirst();                  // 1   (was list.get(0))
bw.getLast();                   // 3   (was list.get(list.size() - 1))
bw.reversed();                  // [3, 2, 1]  (a reversed VIEW, no copy)

List<Integer> ar = new ArrayList<>(List.of(1, 2));
ar.addFirst(0);                 // [0, 1, 2]
ar.removeLast();                // [0, 1]
```

On older judges (Java 17 and below) these don't exist on `List`, so fall back to `get(0)` / `get(size() - 1)`.

## The Collections Utility Class

Static helper methods that work on any `List` (or `Collection`).

```java showLineNumbers
List<Integer> c = new ArrayList<>(List.of(3, 1, 2));
Collections.sort(c);                        // [1, 2, 3]   (same as c.sort(null))
Collections.max(c);                         // 3           O(n)
Collections.min(c);                         // 1           O(n)
Collections.reverse(c);                     // [3, 2, 1]   in place
Collections.sort(c, Collections.reverseOrder());   // [3, 2, 1]  descending
c.sort(Comparator.reverseOrder());          // same thing, List method

Collections.frequency(List.of(1, 2, 2, 3, 2), 2);  // 3   O(n)

List<Integer> so = List.of(1, 2, 3);        // must be SORTED first
Collections.binarySearch(so, 2);            // 1
Collections.binarySearch(so, 5);            // -4   → insertion point = -(-4) - 1 = 3
```

| Method | Example | Result |
| --- | --- | --- |
| `swap(list, i, j)` | `swap([1,2,3,4], 0, 3)` | `[4, 2, 3, 1]` |
| `rotate(list, k)` | `rotate([1,2,3,4,5], 2)` | `[4, 5, 1, 2, 3]` (right by k; negative k rotates left) |
| `nCopies(n, x)` | `nCopies(3, 0)` | `[0, 0, 0]` (immutable; wrap in `new ArrayList<>(...)`) |
| `fill(list, x)` | `fill([1,2,3], 7)` | `[7, 7, 7]` |
| `addAll(coll, a, b, c)` | `addAll(empty, 5, 6, 7)` | `[5, 6, 7]` |
| `shuffle(list, rnd)` | `shuffle([1..5], new Random(42))` | `[2, 3, 4, 5, 1]` (randomised algorithms) |
| `max(coll, comparator)` | longest of `[pear, fig, banana]` by length | `banana` |
| `disjoint(a, b)` | `disjoint([1,2], [3,4])` | `true` (no common elements) |
| `emptyList()`, `singletonList(x)` | | `[]`, `[9]` (immutable) |
| `unmodifiableList(l)` | | read-only view |
| `reverseOrder()` | | a comparator for descending order |

## Legacy: Vector and Stack

```java showLineNumbers
Vector<String> v = new Vector<>();      // like ArrayList but every method synchronized
v.add("Red");
v.add("Blue");                          // [Red, Blue]

Stack<Integer> st = new Stack<>();      // LIFO, extends Vector
st.push(1); st.push(2); st.push(3);
st;                                     // [1, 2, 3]  (printed bottom → top!)
st.peek();                              // 3
st.pop();                               // 3
st.search(1);                           // 2  (1-based distance from the top; -1 if absent)
st.empty();                             // false  (isEmpty() also works)
```

Prefer `ArrayDeque` for stacks: see [Stacks, Queues & Deques](./05-stack-queue-deque.md).

## Gotchas

:::danger[== on Integer in a list]
```java showLineNumbers
List<Integer> big = List.of(1000, 1000);
// error-next-line
big.get(0) == big.get(1);               // ❌ false (two different Integer objects)
big.get(0).equals(big.get(1));          // ✅ true
List.of(100, 100).get(0) == List.of(100, 100).get(1);   // true, only because -128..127 are cached
```
Same trap with `stack.peek() == map.get(k)` and similar. Unbox explicitly (`int a = list.get(i);`) or use `equals`.
:::

- `List.equals` compares elements in order, whatever the implementation: `List.of(1, 2).equals(Arrays.asList(1, 2))` is `true`.
- `Collections.sort` on a `List<int[]>` needs a comparator (arrays aren't `Comparable`).
- `list.toString()` gives `[a, b, c]`, which is handy for debugging; arrays need `Arrays.toString`.
