#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1 CUDA_DEVICE_ORDER=PCI_BUS_ID && python -u ${@:2}