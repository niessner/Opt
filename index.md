---
layout: default
---

Opt
===

Many graphics and vision problems are naturally expressed as optimizations with either linear or non-linear least squares objective functions over visual data, such as images and meshes.
The mathematical descriptions of these functions are extremely concise, but their implementation in real code is tedious, especially when optimized for real-time performance in interactive applications.

We propose a new language, \OPT, in which a user simply writes energy functions over image- or graph-structured unknowns, and a compiler automatically generates state-of-the-art GPU optimization kernels.
The end result is a system in which real-world energy functions in graphics and vision applications are expressible in tens of lines of code. They compile directly into highly optimized GPU solver implementations with performance competitive with the best published hand-tuned, application-specific GPU solvers, and 1--2 orders of magnitude beyond a general-purpose auto-generated solver.
