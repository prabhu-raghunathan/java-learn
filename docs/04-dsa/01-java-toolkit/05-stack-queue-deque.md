---
title: "Stacks, Queues & Deques"
sidebar_position: 5
---

:::tip[TL;DR]
Use **`ArrayDeque`** for all three: as a stack (`push` / `pop` / `peek`), as a FIFO queue (`offer` / `poll` / `peek`), and as a double-ended queue for sliding windows (`offerFirst/Last`, `pollFirst/Last`, `peekFirst/Last`). Every one of those is O(1). It is faster than `Stack` (synchronized) and `LinkedList` (a node per element).
:::

## How the Pieces Fit

| Interface | Meaning | Implementations |
| --- | --- | --- |
| `Queue<E>` | FIFO: add at the tail, remove from the head | `ArrayDeque`, `LinkedList`, `PriorityQueue` (priority order, not FIFO) |
| `Deque<E>` ("deck") | add/remove at **both** ends; can act as a stack or a queue | `ArrayDeque`, `LinkedList` |
| `Stack<E>` (class) | legacy LIFO | `Stack` (extends `Vector`) |

Java has no class called `ArrayQueue`. A queue backed by an array is exactly what **`ArrayDeque`** is (a circular buffer that grows when full).

## ArrayDeque as a Stack (LIFO)

```java showLineNumbers
Deque<Integer> st = new ArrayDeque<>();
st.push(1);
st.push(2);
st.push(3);
st;                         // [3, 2, 1]  (printed top → bottom)
st.peek();                  // 3  look at the top
st.pop();                   // 3  remove the top
st;                         // [2, 1]
st.size();                  // 2
st.isEmpty();               // false
```

`push` = `addFirst`, `pop` = `removeFirst`, `peek` = `peekFirst`. The stack's **top is the head (first)** of the deque.

## ArrayDeque as a Queue (FIFO)

```java showLineNumbers
Queue<Integer> q = new ArrayDeque<>();
q.offer(1);
q.offer(2);
q.offer(3);
q;                          // [1, 2, 3]
q.peek();                   // 1  look at the front
q.poll();                   // 1  remove the front
q;                          // [2, 3]
```

Using `LinkedList` as the queue works the same way and appears in lots of solutions:

```java showLineNumbers
Queue<String> queue = new LinkedList<>();
queue.add("First");
queue.add("Second");
System.out.println(queue.poll());     // First
```

## The Two Method Families: Throw vs Return Special Value

Every queue operation comes in two versions. In DSA the **"returns null/false"** ones are usually more convenient.

| Operation | Throws on failure | Returns `null` / `false` |
| --- | --- | --- |
| Insert at tail | `add(x)` | `offer(x)` |
| Remove head | `remove()` | `poll()` |
| Look at head | `element()` | `peek()` |

```java showLineNumbers
Queue<Integer> e = new ArrayDeque<>();
e.poll();                   // null
e.peek();                   // null
// error-start
e.remove();                 // ❌ NoSuchElementException
e.element();                // ❌ NoSuchElementException
// error-end

Deque<Integer> es = new ArrayDeque<>();
// error-next-line
es.pop();                   // ❌ NoSuchElementException  (pop on an empty Deque)

Stack<Integer> os = new Stack<>();
// error-next-line
os.pop();                   // ❌ EmptyStackException      (pop on an empty Stack)
```

Always guard with `!st.isEmpty()` before `pop()`.

## Full Deque Method Map

| | Head (first) | Tail (last) |
| --- | --- | --- |
| Insert | `offerFirst(x)` / `addFirst(x)` / `push(x)` | `offerLast(x)` / `addLast(x)` / `offer(x)` / `add(x)` |
| Remove | `pollFirst()` / `removeFirst()` / `poll()` / `pop()` | `pollLast()` / `removeLast()` |
| Examine | `peekFirst()` / `getFirst()` / `peek()` | `peekLast()` / `getLast()` |

```java showLineNumbers
Deque<Integer> d = new ArrayDeque<>();
d.offerLast(1);
d.offerLast(2);
d.offerFirst(0);            // [0, 1, 2]
d.peekFirst();              // 0
d.peekLast();               // 2
d.pollFirst();              // 0
d.pollLast();               // 2
d;                          // [1]
```

Other useful methods: `descendingIterator()` (tail → head), `removeFirstOccurrence(x)` (O(n)), `contains(x)` (O(n)).

:::analogy[A deque is a train with doors at both ends]
Passengers can board and leave from the front carriage or the back carriage, but not from the middle. A stack only ever uses the front door; a queue boards at the back and leaves from the front.
:::

## Stack vs ArrayDeque

| | `Stack` | `ArrayDeque` |
| --- | --- | --- |
| Synchronized | Yes (slower) | No |
| Allows `null` | Yes | **No** (`NullPointerException`) |
| Empty `pop()` | `EmptyStackException` | `NoSuchElementException` |
| Print / iteration order | bottom → top | **top → bottom** |
| Index access (`get(i)`) | Yes (it's a `Vector`) | No |

```java showLineNumbers
Stack<Integer> s1 = new Stack<>();          // push 1, 2, 3
Deque<Integer> s2 = new ArrayDeque<>();     // push 1, 2, 3
s1;                                         // [1, 2, 3]
s2;                                         // [3, 2, 1]
new ArrayList<>(s2);                        // [3, 2, 1]  ← if you build a result from a stack, mind the order
```

:::warning[ArrayDeque rejects null]
`new ArrayDeque<Integer>().offer(null)` throws `NullPointerException`, because `poll()` uses `null` to mean "empty". In tree BFS, check `if (node.left != null)` before offering. If you truly must queue `null` (e.g. serialising a tree), use `LinkedList`.
:::

## Patterns

### Balanced parentheses (stack)

```java title="ValidParentheses.java" showLineNumbers
static boolean valid(String s) {
    Deque<Character> st = new ArrayDeque<>();
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else {
            // highlight-next-line
            if (st.isEmpty()) return false;              // closing with nothing open
            char o = st.pop();
            if ((c == ')' && o != '(') || (c == ']' && o != '[') || (c == '}' && o != '{')) return false;
        }
    }
    return st.isEmpty();                                 // leftovers = unclosed
}
// valid("({[]})") → true, valid("(]") → false, valid("((") → false
```

### Monotonic stack (next greater element)

Keep **indices** on the stack whose answer is still unknown, in decreasing order of value. When a bigger value arrives, it is the answer for everything smaller on top.

```java title="NextGreater.java" showLineNumbers
static int[] nextGreater(int[] a) {
    int n = a.length;
    int[] res = new int[n];
    Arrays.fill(res, -1);                                // -1 = no greater element
    Deque<Integer> st = new ArrayDeque<>();              // indices
    for (int i = 0; i < n; i++) {
        // highlight-next-line
        while (!st.isEmpty() && a[st.peek()] < a[i]) res[st.pop()] = a[i];
        st.push(i);
    }
    return res;
}
// nextGreater({2, 1, 2, 4, 3}) → [4, 2, 4, -1, -1]
```

Each index is pushed and popped at most once, so the whole thing is **O(n)**. Store `i - j` instead of `a[i]` and you have "Daily Temperatures": `{73, 74, 75, 71, 69, 72, 76, 73}` → `[1, 1, 4, 2, 1, 1, 0, 0]`.

### BFS on a grid (queue)

```java title="GridBfs.java" showLineNumbers
static int shortestPath(int[][] g) {                     // 0 = open, 1 = wall
    int R = g.length, C = g[0].length;
    int[][] dist = new int[R][C];
    for (int[] row : dist) Arrays.fill(row, -1);         // -1 = not visited
    Deque<int[]> q = new ArrayDeque<>();
    q.offer(new int[]{0, 0});
    dist[0][0] = 0;
    int[] dr = {-1, 1, 0, 0}, dc = {0, 0, -1, 1};
    while (!q.isEmpty()) {
        int[] cur = q.poll();
        int r = cur[0], c = cur[1];
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr < 0 || nc < 0 || nr >= R || nc >= C || g[nr][nc] == 1 || dist[nr][nc] != -1) continue;
            // highlight-next-line
            dist[nr][nc] = dist[r][c] + 1;               // mark visited WHEN ENQUEUED, not when polled
            q.offer(new int[]{nr, nc});
        }
    }
    return dist[R - 1][C - 1];
}
// grid {{0,0,1},{1,0,1},{0,0,0}} → 4
```

For level-by-level processing (tree level order), snapshot the size first: `for (int k = q.size(); k > 0; k--) { ... q.poll() ... }`.

### Sliding window maximum (monotonic deque)

The deque holds indices whose values are **decreasing** from head to tail. The head is always the max of the current window.

```java title="SlidingWindowMax.java" showLineNumbers
static int[] maxSlidingWindow(int[] a, int k) {
    int n = a.length;
    int[] res = new int[n - k + 1];
    Deque<Integer> dq = new ArrayDeque<>();              // indices
    for (int i = 0; i < n; i++) {
        if (!dq.isEmpty() && dq.peekFirst() <= i - k) dq.pollFirst();       // fell out of the window
        while (!dq.isEmpty() && a[dq.peekLast()] <= a[i]) dq.pollLast();    // smaller ones are useless now
        dq.offerLast(i);
        // highlight-next-line
        if (i >= k - 1) res[i - k + 1] = a[dq.peekFirst()];
    }
    return res;
}
// maxSlidingWindow({1, 3, -1, -3, 5, 3, 6, 7}, 3) → [3, 3, 5, 5, 6, 7]
```

## Gotchas

- `Deque<Integer> st` + `st.peek() == other.peek()` compares `Integer` objects: wrong above 127. Unbox into `int` variables first.
- `Queue<Integer> q = new PriorityQueue<>()` is a `Queue` too, but `poll()` returns the **smallest**, not the oldest. See [Priority Queue & Heaps](./06-priority-queue-heap.md).
- Mark BFS nodes visited **when you enqueue them**, not when you dequeue; otherwise the same node gets queued many times.
- `ArrayDeque` has no `get(i)`. If you need random access plus stack behaviour, use an `ArrayList` and `remove(size() - 1)`.
