"""
Builds benchmark/queries.jsonl: v1's annotated queries plus new ones.

    python benchmark/new_queries.py

v1's 93 queries are mostly "explain <textbook topic>". The new ones add what
real students ask: comparisons, "why" questions, specific mechanisms from
later lectures, and questions these 9 courses do NOT cover. Those
out-of-scope questions exist so a future answer generator can be scored on
saying "not covered" instead of inventing an answer. Whether a question is
covered is decided by relevance labels, not by this file; `intent` only
records what the question was written to test.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEW = {
    "cn": [
        "why does TCP use a three way handshake instead of two",
        "difference between CIDR and classful IP addressing",
        "how does BGP choose routes between autonomous systems",
        "what happens to the TCP congestion window when a packet is lost",
        "how does DHCP give an IP address to a new device",
        "why does wifi use CSMA/CA instead of CSMA/CD",
        "how to write a TCP server with socket bind listen accept",
    ],
    "coa": [
        "what does Amdahl's law say about the speedup from improving one part",
        "difference between hardwired and microprogrammed control unit",
        "how do data hazards happen in a pipeline and how does forwarding fix them",
        "how does DMA move data without involving the CPU",
        "direct mapped vs set associative cache",
        "how is a floating point number stored in IEEE 754 format",
        "how does Booth's algorithm multiply signed numbers",
    ],
    "cv": [
        "why is the Harris corner detector rotation invariant",
        "what is the fundamental matrix and the epipolar constraint",
        "how does bundle adjustment refine camera poses and 3D points",
        "difference between SIFT and SURF descriptors",
        "how is the Hough transform used to detect lines",
        "what are intrinsic vs extrinsic camera parameters",
        "how do region based CNNs like R-CNN detect objects",
    ],
    "daa": [
        "why does Dijkstra's algorithm fail with negative edge weights",
        "how does union find with path compression work",
        "how to model a production planning problem as a linear program",
        "max flow min cut theorem and the Ford Fulkerson method",
        "longest common subsequence dynamic programming table",
        "why is the greedy choice correct for interval scheduling",
        "how does Huffman coding build an optimal prefix code",
    ],
    "dbms": [
        "what is the difference between 2NF 3NF and BCNF",
        "how does a B+ tree handle insertion and node splits",
        "what is a correlated subquery in SQL",
        "undo logging vs redo logging for crash recovery",
        "when is a schedule conflict serializable",
        "lossless join decomposition and dependency preservation",
        "how does the query optimizer choose a join order",
    ],
    "dl": [
        "why does momentum help gradient descent converge faster",
        "how does backpropagation compute gradients layer by layer",
        "what is the link between PCA and autoencoders",
        "skip-gram vs continuous bag of words in word2vec",
        "how does hierarchical softmax reduce computation",
        "why use unsupervised pre-training before training a deep network",
        "guided backpropagation to visualise what a CNN has learned",
        "how does early stopping act as a regulariser",
    ],
    "dsa": [
        "how to find the maximum sum subarray efficiently",
        "how to check whether a graph is bipartite using BFS",
        "what is an articulation point in a graph and how to find it",
        "how does a red black tree stay balanced",
        "how to build a heap in linear time",
        "how to find the k-th smallest element without sorting",
    ],
    "ml": [
        "what is inductive bias in machine learning",
        "how does collaborative filtering recommend items",
        "what is the kernel trick in SVM",
        "soft margin SVM with slack variables",
        "how does bagging reduce variance in an ensemble",
        "single vs complete linkage in hierarchical clustering",
        "sample complexity for a finite hypothesis space",
    ],
    "os": [
        "what happens during a context switch in xv6",
        "how does the MMU translate a virtual address to a physical address",
        "how does the Linux CFS scheduler pick the next process",
        "Peterson's solution to the critical section problem",
        "how does a test and set spinlock work",
        "how does a buffer overflow attack overwrite the return address",
        "difference between a process and a thread",
    ],
}

# Computer-science questions near these courses but (most likely) not in them.
OUT_OF_SCOPE = [
    "how does bitcoin proof of work reach consensus",
    "explain the Raft consensus algorithm leader election",
    "what is the CAP theorem for distributed databases",
    "how does garbage collection work in the Java virtual machine",
    "how does the React virtual DOM diffing work",
    "how is RLHF used to fine tune large language models",
    "explain Shor's algorithm for factoring with a quantum computer",
    "what is a Kubernetes pod and how is it scheduled",
    "how does git rebase differ from git merge",
    "how do compilers do register allocation by graph colouring",
    "how does GPS work out a receiver's position",
    "what is the difference between OAuth and OpenID Connect",
]


def main() -> None:
    rows = []
    for line in (ROOT / "data/eval/annotations.jsonl").read_text().splitlines():
        a = json.loads(line)
        if a.get("status") == "selected":
            rows.append({"qid": a["id"], "query": a["query"], "course": a["expected_course"],
                         "intent": "v1-seed", "type": a.get("type")})
    for course, queries in NEW.items():
        for i, q in enumerate(queries, 1):
            rows.append({"qid": f"{course}_n{i:02d}", "query": q, "course": course,
                         "intent": "student-doubt", "type": None})
    for i, q in enumerate(OUT_OF_SCOPE, 1):
        rows.append({"qid": f"oos_{i:02d}", "query": q, "course": None,
                     "intent": "out-of-scope", "type": None})
    assert len({r["qid"] for r in rows}) == len(rows)
    out = ROOT / "benchmark" / "queries.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} queries → {out}")


if __name__ == "__main__":
    main()
