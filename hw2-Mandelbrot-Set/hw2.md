# Homework 2: Mandelbrot Set
## Goal
This assignment helps you get familiar with:

Pthread

Hybrid parallelism (MPI + OpenMP)

Load balancing techinques

In this assignment, you are asked to parallelize the sequential Mandelbrot Set program using:

(hw2a) – pthread / std::thread

(hw2b) – MPI + OpenMP

## Problem Description
Mandelbrot Set is a set of complex numbers that are quasi-stable when computed by iterating the function:
```
{Z0Zk=C=Z2k−1+C
C is some complex number: C=a+bi
```
Zk+1 is the (k+1)th iteration of the complex number

if |Zk|≤2 for any k, C belongs to Mandelbrot Set