---
name: research-method
description: Researches about a new method (scalarizer or aggregator) from the scientific literature and how to integrate it to TorchJD. Use when a contributor is interested into adding a new aggregator, scalarizer, or a new method that could be either, that is not already listed in the tracking issues.
---

# Research new method

This skill researches about a new method by reading the paper and the existing implementations, proposing a plan for integrating an implmentation in TorchJD.

**For agents:** invoke as `/implement-method method-name (paper-name)` (e.g. `/implement-method upgrad (Jacobian Descent for Multi-objective Optimization)`).
If no method name is provided, ask the user for the name of the method and the title of the paper.

**For humans:** follow the numbered steps below to guide you in your development.

---

## Instructions

### Step 1: Generate a tracking summary

On GitHub, we have a tracking issue for scalarizers (https://github.com/SimplexLab/TorchJD/issues/667) and one for aggregators (https://github.com/SimplexLab/TorchJD/issues/665).
The first goal is to generate a new row for the table in the appropriate tracking issue.
You will also have to find the official implementation (if any), that could be linked in the paper, and the most known non-official implementations (search in particular in LibMTL, libmoon and pymoo, and search for more repos online). You will have to find the exact file(s) and lines in which the method is implemented.
You can also write special remarks if anything deserves further attention / if you're not sure about something.
Do not bother trying to find citation count or the exact venue in which the paper was published, unless you directly see this information from your initial search.

### Step 2: Analyze the method

Based on the paper and the existing implementations, find the things in the interface of the new method that are non-standard compared to existing methods. For example, is the method stateful (and in which way), random, does it require some warm-up period, other stats than those provided through the forward method (like losses for aggregators), and so on. Report all of those non-standard things.

### Step 3: Produce summary row

Produce and output a row that can be directly copy-pasted to this table, but do not modify the issue yourself. Propose to post a comment on the issue with the row.


---
