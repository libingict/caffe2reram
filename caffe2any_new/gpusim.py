import sys
import argparse
import caffe_pb2 as caffe
from google.protobuf import text_format
import math
import numpy as np
import ntpath
import random

#For Fermi Architecture, one cycle = 1/7*10-6s
def lenet(vias):
	if vias == '0':
		lenet_FF=[{'latency': 51426.719999999994, 'energy': 2540479.2986119995, 'name': 'conv1'}, {'latency': 5029.639999999999, 'energy': 213349.28137599997, 'name': 'pool1'}, {'latency': 255071.76, 'energy': 12732714.16259, 'name': 'conv2'}, {'latency': 3135.3599999999997, 'energy': 121788.669696, 'name': 'pool2'}, {'latency': 289743.9, 'energy': 11186738.334492002, 'name': 'fc1'}, {'latency': 1559.1599999999999, 'energy': 70964.855568, 'name': 'relu1'}, {'latency': 74931.98, 'energy': 2591257.4005259997, 'name': 'fc2'}]
		lenet_FF_BP=[{'latency': 51553.09999999999, 'name': 'conv1', 'energy': 2542870.8731199997, 'BPenergy': 31088431.705163997, 'BPlatency': 856092.4400000001, 'G_energy': 15140844.421337998, 'G_latency': 417972.74}, {'latency': 5089.28, 'energy': 216847.095808, 'name': 'pool1', 'BPenergy': 239767.42032, 'BPlatency': 4759.84}, {'latency': 256585.47999999998, 'name': 'conv2', 'energy': 12790031.362515999, 'BPenergy': 9927440.192868, 'BPlatency': 229554.36, 'G_energy': 3113026.397904, 'G_latency': 72918.42}, {'latency': 3072.8799999999997, 'energy': 122208.43759999999, 'name': 'pool2', 'BPenergy': 204227.88212399997, 'BPlatency': 5675.74}, {'latency': 289846.14, 'name': 'fc1', 'energy': 11189489.981808001, 'BPenergy': 34878834.89683, 'BPlatency': 408355.08, 'G_energy': 3794747.485564, 'G_latency': 46232.36}, {'latency': 1588.98, 'energy': 71033.76192, 'name': 'relu1', 'BPenergy': 66482.664192, 'BPlatency': 1908.48}, {'latency': 74931.98, 'name': 'fc2', 'energy': 2595646.68755, 'BPenergy': 767613.4776720001, 'BPlatency': 12112.599999999999, 'G_energy': 104526.37948799999, 'G_latency': 1951.08}]
	elif vias == '1':	
		lenet_FF_BP=[{'latency': 48116.7, 'name': 'conv1', 'energy': 2523091.2397140004, 'BPenergy': 31261269.993979998, 'BPlatency': 861795.16, 'G_energy': 15225965.552976, 'G_latency': 420959.0}, {'latency': 4934.5, 'energy': 210171.70435, 'name': 'pool1', 'BPenergy': 230731.724742, 'BPlatency': 4695.94}, {'latency': 246115.81999999998, 'name': 'conv2', 'energy': 12676760.55146, 'BPenergy': 9809422.720689999, 'BPlatency': 223384.46, 'G_energy': 2955475.6219099998, 'G_latency': 69801.51999999999}, {'latency': 3105.54, 'energy': 123336.52110000001, 'name': 'pool2', 'BPenergy': 203819.35494, 'BPlatency': 5672.9}, {'latency': 289129.04, 'name': 'fc1', 'energy': 10911818.70682, 'BPenergy': 34706766.773555994, 'BPlatency': 399954.36, 'G_energy': 3749490.6320519997, 'G_latency': 42487.82}, {'latency': 1557.74, 'energy': 69315.068328, 'name': 'relu1', 'BPenergy': 61522.082200000004, 'BPlatency': 1766.48}, {'latency': 74842.52, 'name': 'fc2', 'energy': 2584999.579664, 'BPenergy': 776313.354752, 'BPlatency': 11922.32, 'G_energy': 101865.07271400001, 'G_latency': 1912.74}]
	
		lenet_FF=[{'latency': 48116.7, 'energy': 2523091.2397140004, 'name': 'conv1'}, {'latency': 4921.719999999999, 'energy': 209568.80628799996, 'name': 'pool1'}, {'latency': 247094.19999999998, 'energy': 12706226.510669999, 'name': 'conv2'}, {'latency': 3054.42, 'energy': 118673.38026, 'name': 'pool2'}, {'latency': 289131.88, 'energy': 10182767.011848, 'name': 'fc1'}, {'latency': 1557.74, 'energy': 71165.35190000001, 'name': 'relu1'}, {'latency': 74842.52, 'energy': 2587156.7547720005, 'name': 'fc2'}]
	else:
		lenet_FF_BP=[{'latency': 42830.03999999999, 'name': 'conv1', 'energy': 2466025.035154, 'BPenergy': 29426890.264076, 'BPlatency': 807085.4, 'G_energy': 14315311.720965998, 'G_latency': 393771.68}, {'latency': 4680.32, 'energy': 206827.085056, 'name': 'pool1', 'BPenergy': 220495.655736, 'BPlatency': 4524.12}, {'latency': 228395.63999999998, 'name': 'conv2', 'energy': 12414266.543475999, 'BPenergy': 9457806.563508, 'BPlatency': 210997.8, 'G_energy': 2850801.189672, 'G_latency': 63811.95999999999}, {'latency': 2936.56, 'energy': 117683.22931200001, 'name': 'pool2', 'BPenergy': 184660.47666400002, 'BPlatency': 5072.24}, {'latency': 243267.3, 'name': 'fc1', 'energy': 9400020.417953998, 'BPenergy': 34583137.193992, 'BPlatency': 392601.6, 'G_energy': 3719240.030116, 'G_latency': 40279.72}, {'latency': 1314.9199999999998, 'energy': 59913.146371999996, 'name': 'relu1', 'BPenergy': 53719.12469, 'BPlatency': 1540.6999999999998}, {'latency': 66132.24, 'name': 'fc2', 'energy': 2288559.948694, 'BPenergy': 757310.866472, 'BPlatency': 11459.399999999998, 'G_energy': 115316.32588399999, 'G_latency': 1847.4199999999998}]
		lenet_FF=[{'latency': 42861.28, 'energy': 2466863.9710119995, 'name': 'conv1'}, {'latency': 4507.08, 'energy': 201287.094216, 'name': 'pool1'}, {'latency': 229172.37999999998, 'energy': 12416532.360654, 'name': 'conv2'}, {'latency': 2923.7799999999997, 'energy': 117852.016618, 'name': 'pool2'}, {'latency': 243203.4, 'energy': 8986743.479646, 'name': 'fc1'}, {'latency': 1314.9199999999998, 'energy': 60484.742095999994, 'name': 'relu1'}, {'latency': 66132.24, 'energy': 2286946.611576, 'name': 'fc2'}]
	return lenet_FF_BP

def alexnet(vias):
	pass

def vgg(vias):
	pass

def get_network(network,vias):
	if network.find('lenet')!=-1:
		return lenet(vias)
	elif network.find('alexnet')!=-1:
		return alexnet(vias)
	elif network.find('vgg')!=-1:
		return vgg(vias)
"""
net=lenet(0)
for layer in net:
	for key, value in layer.items():
		print(key, value)
"""		

