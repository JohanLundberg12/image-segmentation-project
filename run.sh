#!/bin/sh

echo "Running training scenarios ..."

echo "Starting 00 ..."
python train.py configs/baseline_config_00.yaml
echo "####################################################"
echo "Training 00 - Done"

echo "Starting 01 ..."
python train.py configs/baseline_config_01.yaml
echo "####################################################"
echo "Training 01 - Done"

echo "Starting 02 ..."
python train.py configs/baseline_config_02.yaml
echo "####################################################"
echo "Training 02 - Done"

echo "Starting 03 ..."
python train.py configs/baseline_config_03.yaml
echo "####################################################"
echo "Training 03 - Done"

echo "Starting 04 ..."
python train.py configs/baseline_config_04.yaml
echo "####################################################"
echo "Training 04 - Done"

echo "Starting 05 ..."
python train.py configs/baseline_config_05.yaml
echo "####################################################"
echo "Training 05 - Done"

echo "Starting 06 ..."
python train.py configs/baseline_config_06.yaml
echo "####################################################"
echo "Training 06 - Done"

