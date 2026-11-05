"""One-pass ordering and explicit optimizer-step learning-rate semantics.

Integrated from the former Experiments/020-one-epoch-training/epoch_core.py (tag
archive/experiments-011-021 in the workspace repository). Logic unchanged.
"""
import math,random

def epoch_order(entries,seed,smoke_rows=0):
    if not entries or smoke_rows<0:raise ValueError('invalid row pool')
    order=list(range(len(entries)));random.Random(seed).shuffle(order)
    longest=max(order,key=lambda i:entries[i]['sequence_tokens']);order.remove(longest);order.insert(0,longest)
    if smoke_rows:
        if smoke_rows>len(entries):raise ValueError('smoke exceeds dataset')
        # Stress longest sequence, then cheap rows; separate from full-epoch quality runs.
        order=[longest]+sorted((i for i in order if i!=longest),key=lambda i:(entries[i]['sequence_tokens'],i))[:smoke_rows-1]
    return order

def groups(order,accumulation):
    if accumulation<1 or len(set(order))!=len(order):raise ValueError('invalid accumulation/order')
    return [order[i:i+accumulation] for i in range(0,len(order),accumulation)]

def lr_factor(index,total,warmup,min_ratio=.1):
    if not 0<=index<total or not 0<=warmup<total or not 0<=min_ratio<=1:raise ValueError('invalid schedule')
    if index<warmup:return (index+1)/warmup
    progress=(index-warmup)/max(1,total-warmup-1)
    return min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*progress))

def remaining_rows(order,completed,accumulation):
    chunks=groups(order,accumulation)
    if not 0<=completed<=len(chunks):raise ValueError('invalid completed cursor')
    return [i for chunk in chunks[completed:] for i in chunk]
