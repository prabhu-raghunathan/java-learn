---
title: "HashSet & HashMap (plus Linked variants)"
sidebar_position: 7
---

:::tip[TL;DR]
`HashSet` answers "have I seen this?" and `HashMap` answers "what is stored for this key?", both in **O(1) average**. They are the most used structures in DSA: frequency counts, two-sum lookups, visited sets, grouping, prefix-sum counts. Learn the modern map methods (`getOrDefault`, `merge`, `computeIfAbsent`) and your counting code shrinks to one line.
:::

## How Hashing Works (30-second version)

| Step | What happens |
| --- | --- |
| 1 | `key.hashCode()` gives an `int` |
| 2 | The map mixes the high bits in (`h ^ (h >>> 16)`) and picks a bucket: `hash & (capacity - 1)` |
| 3 | Inside the bucket, keys are compared with `equals()` |
| 4 | When size exceeds `capacity × 0.75` (the **load factor**), the table doubles and every entry is re-bucketed |
| 5 | A bucket with 8+ colliding entries (in a table of 64+ buckets) becomes a red-black tree, so lookups there stay O(log n) |

That is why a custom key class needs correct `equals` and `hashCode` (see [Collections Overview](./01-collections-overview.md#hash-based-collections-need-equals--hashcode)).

## HashSet

```java showLineNumbers
Set<Integer> s = new HashSet<>();
s.add(5);                       // true   (added)
s.add(5);                       // false  (already there: add tells you!)
s.contains(5);                  // true
s.remove(5);                    // true   (was there)
s.remove(5);                    // false
s.size();                       // 0
```

The boolean from `add` makes duplicate detection a one-liner:

```java showLineNumbers
int[] arr = {3, 1, 3, 2};
Set<Integer> seen = new HashSet<>();
for (int v : arr) {
    // highlight-next-line
    if (!seen.add(v)) System.out.println("dup " + v);     // dup 3
}

new HashSet<>(Arrays.asList(3, 1, 3, 2)).size();          // 3 distinct values
```

### Set algebra

The bulk methods **modify the set they are called on**, so copy first.

```java showLineNumbers
Set<Integer> a = new HashSet<>(List.of(1, 2, 3, 4));
Set<Integer> b = new HashSet<>(List.of(3, 4, 5));

Set<Integer> union = new HashSet<>(a);  union.addAll(b);      // [1, 2, 3, 4, 5]
Set<Integer> inter = new HashSet<>(a);  inter.retainAll(b);   // [3, 4]
Set<Integer> diff  = new HashSet<>(a);  diff.removeAll(b);    // [1, 2]
a.containsAll(Set.of(1, 2));                                  // true (subset test)
```

## HashMap

```java showLineNumbers
Map<String, Integer> m = new HashMap<>();
m.put("a", 1);                  // null   (returns the PREVIOUS value)
m.put("a", 2);                  // 1      (overwrote it)
m.get("a");                     // 2
m.get("zz");                    // null   (missing key)
m.getOrDefault("zz", 0);        // 0
m.containsKey("a");             // true   O(1)
m.containsValue(2);             // true   O(n)! scans every entry
m.remove("a");                  // 2      (returns the removed value)
m.size();                       // 0
```

### Counting frequencies: three ways

```java showLineNumbers
Map<Character, Integer> f = new HashMap<>();
for (char c : "banana".toCharArray()) {
    // classic
    f.put(c, f.getOrDefault(c, 0) + 1);
}
// {a=3, b=1, n=2}

Map<Character, Integer> f2 = new HashMap<>();
// highlight-next-line
for (char c : "banana".toCharArray()) f2.merge(c, 1, Integer::sum);   // modern one-liner
// {a=3, b=1, n=2}
```

`merge(key, value, fn)`: if the key is absent, store `value`; otherwise store `fn(old, value)`. **If `fn` returns `null`, the key is removed**, which is perfect for sliding windows where a count drops to zero:

```java showLineNumbers
f2.merge('n', -2, (o, n) -> o + n == 0 ? null : o + n);   // n's count hits 0 → key removed
// {a=3, b=1}  (map.size() is now the number of DISTINCT chars in the window)
```

For lowercase letters only, an `int[26]` is still faster than any map.

### The modern map methods

| Method | Does | Typical use |
| --- | --- | --- |
| `getOrDefault(k, d)` | value or `d` if missing | counting, safe lookups |
| `merge(k, v, fn)` | insert `v` or combine with `fn` | counting: `merge(k, 1, Integer::sum)` |
| `computeIfAbsent(k, k -> new X())` | create the value if missing, **return it** | adjacency lists, grouping |
| `putIfAbsent(k, v)` | insert only if missing; returns existing value or `null` | first-seen index |
| `compute(k, (k, v) -> ...)` | recompute the value (`v` is `null` if absent) | custom updates |
| `computeIfPresent(k, (k, v) -> ...)` | only if present; returning `null` removes | decrement and remove |
| `entrySet().removeIf(e -> ...)` | remove entries matching a condition | clean-up |

```java showLineNumbers
Map<Integer, List<String>> g = new HashMap<>();
// highlight-next-line
g.computeIfAbsent(1, k -> new ArrayList<>()).add("x");   // creates the list, then adds
g.computeIfAbsent(1, k -> new ArrayList<>()).add("y");   // list exists, just adds
// {1=[x, y]}

Map<String, Integer> pi = new HashMap<>();
pi.putIfAbsent("k", 1);         // null  (inserted)
pi.putIfAbsent("k", 2);         // 1     (kept the old value)
```

### Iterating

```java showLineNumbers
for (Map.Entry<String, Integer> e : map.entrySet()) {     // key + value together (best)
    String k = e.getKey();
    int v = e.getValue();
}
for (String k : map.keySet()) { }                         // keys only
for (int v : map.values()) { }                            // values only
map.forEach((k, v) -> System.out.println(k + "=" + v));   // lambda
for (var e : map.entrySet()) { }                          // var saves typing the entry type
```

Iterating `keySet()` and then calling `map.get(k)` does two lookups per key; `entrySet()` does none.

:::danger[Don't remove inside a for-each over the map]
```java showLineNumbers
// error-next-line
for (String k : map.keySet()) map.remove(k);              // ❌ ConcurrentModificationException
// success-start
map.entrySet().removeIf(e -> e.getValue() % 2 == 1);      // ✅ remove by condition
map.values().removeIf(v -> v == 2);                       // ✅ also works through values()
// success-end
```
:::

## Patterns

### Two Sum (value → index)

```java showLineNumbers
int[] nums = {2, 7, 11, 15};
int target = 9;
Map<Integer, Integer> idx = new HashMap<>();
for (int i = 0; i < nums.length; i++) {
    Integer j = idx.get(target - nums[i]);                // Integer, so null means "not found"
    if (j != null) return new int[]{j, i};                // [0, 1]
    idx.put(nums[i], i);
}
```

### Group Anagrams (key → list)

```java showLineNumbers
String[] strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
Map<String, List<String>> groups = new HashMap<>();
for (String w : strs) {
    char[] c = w.toCharArray();
    Arrays.sort(c);
    groups.computeIfAbsent(new String(c), k -> new ArrayList<>()).add(w);
}
// {abt=[bat], aet=[eat, tea, ate], ant=[tan, nat]}
return new ArrayList<>(groups.values());
```

### Subarray Sum Equals K (prefix sum → count)

```java showLineNumbers
int[] ps = {1, 1, 1};
int k = 2;
Map<Integer, Integer> seen = new HashMap<>();
// highlight-next-line
seen.put(0, 1);                                           // empty prefix: don't forget it
int sum = 0, count = 0;
for (int v : ps) {
    sum += v;
    count += seen.getOrDefault(sum - k, 0);               // how many earlier prefixes complete a sum of k
    seen.merge(sum, 1, Integer::sum);
}
// count = 2
```

### Sorting a map by value

A `HashMap` can't be sorted; copy the entries into a list and sort that.

```java showLineNumbers
Map<String, Integer> sc = new HashMap<>(Map.of("x", 3, "y", 1, "z", 2));
List<Map.Entry<String, Integer>> es = new ArrayList<>(sc.entrySet());
es.sort(Map.Entry.comparingByValue());                    // [y=1, z=2, x=3]
es.sort(Map.Entry.<String, Integer>comparingByValue().reversed());   // [x=3, z=2, y=1]

// stream version: keys ordered by value, descending
sc.entrySet().stream()
  .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
  .map(Map.Entry::getKey)
  .collect(Collectors.toList());                          // [x, z, y]
```

To keep keys sorted all the time, use a `TreeMap` instead: see [TreeMap & TreeSet](./08-treemap-treeset.md).

## LinkedHashSet & LinkedHashMap

Same O(1) operations as the hash versions, plus a doubly linked list through the entries that remembers **insertion order**.

```java showLineNumbers
Set<Character> lhs = new LinkedHashSet<>();
for (char c : "banana".toCharArray()) lhs.add(c);
// [b, a, n]  (first-seen order)   vs HashSet of "zebra": [a, b, r, e, z] (hash order)

Map<String, Integer> lhm = new LinkedHashMap<>();
lhm.put("z", 1); lhm.put("a", 2); lhm.put("m", 3);
// {z=1, a=2, m=3}
```

### First non-repeating character

```java showLineNumbers
Map<Character, Integer> fo = new LinkedHashMap<>();
for (char c : "leetcode".toCharArray()) fo.merge(c, 1, Integer::sum);
for (var e : fo.entrySet()) {
    if (e.getValue() == 1) { System.out.println(e.getKey()); break; }   // l
}
```

### LRU cache in 6 lines

The constructor flag `accessOrder = true` moves an entry to the end every time it is read; `removeEldestEntry` evicts from the front.

```java title="LRU.java" showLineNumbers
class LRU<K, V> extends LinkedHashMap<K, V> {
    private final int cap;
    // highlight-next-line
    LRU(int cap) { super(16, 0.75f, true); this.cap = cap; }      // true = access order
    @Override protected boolean removeEldestEntry(Map.Entry<K, V> e) { return size() > cap; }
}
// cap 2: put 1, put 2, get 1, put 3 → keys [1, 3]  (2 was least recently used)
```

In an interview you may be asked to build it by hand (`HashMap<Integer, Node>` + your own doubly linked list); this version is the answer to "is there a built-in way?".

## Gotchas

:::danger[Comparing two Integer values from maps]
```java showLineNumbers
m1.put("x", 1000);  m2.put("x", 1000);
// error-next-line
m1.get("x") == m2.get("x");                     // ❌ false  (two Integer objects)
m1.get("x").equals(m2.get("x"));                // ✅ true
m1.get("x").intValue() == m2.get("x");          // ✅ true   (one side unboxed → numeric compare)
```
It works for -128..127 because of the `Integer` cache, which is why tests pass on small inputs and fail on large ones.
:::

:::danger[Unboxing a missing key]
```java showLineNumbers
// error-next-line
int bad = m1.get("missing");                    // ❌ NullPointerException (null can't become int)
int ok = m1.getOrDefault("missing", 0);         // ✅
```
:::

- `HashMap` allows one `null` key and `null` values; `TreeMap` throws on a `null` key; `Map.of` / `Set.of` reject `null` entirely.
- `getOrDefault` returns the default only when the key is **absent**. If the key maps to `null`, you get `null`: `{a=null}.getOrDefault("a", 5)` is `null`.
- Iteration order of `HashMap`/`HashSet` is unspecified and can change between runs or Java versions. Never print a `HashMap` as a final answer that needs an order.
- `Map.entry("k", 5)` and `new AbstractMap.SimpleEntry<>("k", 5)` create a standalone key-value pair, handy as a quick tuple.
- Pre-size big maps: `new HashMap<>(2 * n)` avoids repeated resizing when you know you'll insert `n` keys.
