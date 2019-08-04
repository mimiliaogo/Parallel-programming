# Homework 1: Odd-Even Sort
## Goal
This assignment helps you get familiar with MPI by implementing odd-even sort. We encourage you to optimize your program by exploring different parallelizing strategies.

## Problem Description
In this assignment, you are required to implement odd-even sort algorithm using MPI Library under the restriction that MPI process can only send data messages to its neighbor processes. Odd-even sort is a comparison sort which consists of two main phases: even-phase and odd-phase.

In even-phase, all even/odd indexed pairs of adjacent elements are compared. If a pair is in the wrong order, the elements are switched. Similarly, the same process repeats for odd/even indexed pairs in odd-phase. The odd-even sort algorithm works by alternating these two phases until the list is completely sorted.

In order for you to understand this algorithm better, the execution flow of odd-even sort is illustrated step by step as below: (We are sorting the list into ascending order in this case)