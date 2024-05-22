---
title: "Data Science Interviewing Pt. 3: Number of Islands Edition"
date: "2023-08-15"
description: "Data science interviewing is interesting. In this several part series, I will explain/implement solutions to problems that I have come across during the data science interview process. These posts serve as practice, but I hope that others will find them useful as well. This time it's Number of Islands edition."
---

During a past interview, I was subjected to LeetCode challenges. In this post, I'll provide an overview of one of the problems from that interview, which was LeetCode 200. Number of Islands in Python.

## Problem
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**Example 1:**

Input: 
```
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
```
Output: 1

**Example 2:**

Input: 
```
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
```
Output: 3

Constraints:
* m == grid.length
* n == grid[i].length
* 1 <= m, n <= 300
* grid[i][j] is '0' or '1'.

## How does this Relate to Data Science?
The Number of Islands problem relates to data science in the context of graph theory and connectivity analysis, which are important in various data science applications. 

## Depth-first Search Solution


## Breadth-first Search Solution

