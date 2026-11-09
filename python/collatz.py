#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 14:17:08 2023

@author: mcarroll
"""

def collatz_sequence(n, level=0):
    print("  " * level + str(n))  # Display current number with indentation
    
    if n == 1:
        return
    
    if n % 2 == 0:
        collatz_sequence(n // 2, level + 1)
    else:
        collatz_sequence(3 * n + 1, level + 1)

# Example usage
number = int(input("Enter a positive integer: "))
print("Collatz sequence:")
collatz_sequence(number)
