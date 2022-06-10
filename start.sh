#!/bin/bash

echo 'alias python="/usr/bin/python3"' >> ~/.bashrc
echo 'alias mlps3="/apollo/bin/env -e HoverboardDefaultMLPS3Tool mlps3"' >> ~/.bashrc
echo 'alias scale="/apollo/env/HoverboardScaleCLI/bin/hover-scale scale"' >> ~/.bashrc
# Add this in your .bashrc
#alias discover="/usr/bin/curl -sk https://host-discovery.hoverboard | jq -c '.[] | {privateIpAddress: .privateIpAddress, tagName: .tags.Name, instanceType: .instanceType}'"

source ~/.bashrc

# Download data
if [ ! -d /home/ec2-user/raw_data/ ]; then
  mkdir /home/ec2-user/raw_data/
fi

###### data from alexa ######
# raw data
#mlps3 cp -r "s3://hoverboard-shared-rnascience-data-na/programs/self-learning/experiments/14th_ab_constrained_bandit/1_0_0_3_LPRP_MetaGrad_BenchmarkGlobal_v3/knowledge_distill_data_preprocess/hold_out/part-000*" /home/ec2-user/raw_data/preprocess/
# processed data
#mlps3 cp -r "s3://hoverboard-shared-rnascience-data-na/programs/self-learning/experiments/14th_ab_constrained_bandit/1_0_0_3_LPRP_MetaGrad_BenchmarkGlobal_v3/knowledge_distill_data_preprocess/train/part-00*" /home/ec2-user/raw_data/train/
#mlps3 cp -r "s3://hoverboard-shared-rnascience-data-na/programs/self-learning/experiments/14th_ab_constrained_bandit/1_0_0_3_LPRP_MetaGrad_BenchmarkGlobal_v3/knowledge_distill_data_preprocess/valid/part-00*" /home/ec2-user/raw_data/valid/
# model/vocab
#mlps3 cp "s3://hoverboard-shared-rnascience-data-na/prod-models/dar-candle-production/en-us/1.2.0.0/2022-02-16T19:29/RP/model.pth" /home/ec2-user/raw_data/model.pth

###### my data ######
mlps3 cp s3://bluetrain-workspaces/wuti/2022/vocab.json /home/ec2-user/raw_data/vocab.json
mlps3 cp s3://bluetrain-workspaces/wuti/2022/process.ipynb /home/ec2-user/raw_data/process.ipynb
mlps3 cp -r s3://bluetrain-workspaces/wuti/2022/wuti_data/ /home/ec2-user/raw_data/wuti_data/

echo 1. Download raw data complete

# Install relevant packages
pip install -r requirements.txt

echo 2. Install required package