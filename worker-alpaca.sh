#!/bin/bash

celery -A backend.alpaca worker --loglevel=INFO --concurrency 1 --pool threads --hostname "alpaca$CUDA_VISIBLE_DEVICES@%h"