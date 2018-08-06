import sys
import argparse
import caffe_pb2 as caffe
from google.protobuf import text_format
import numpy as np
import ntpath
import gpusim
import parse_caffe
from revise_parse import parse_input, tile_design, wires_overhead,cost
#file_name = "examples/cifar10_full.prototxt"
#file_name = "examples/alexnet.deploy.prototxt"


def get_wire_overhead(network,vias,bit_width=1):
	return wires_overhead(network,vias,bit_width=1)
def get_reram(network,train,cells):
	reram_overhead=tile_design(network)
	cost(reram_overhead,train,cells)
	return reram_overhead
def get_gpu(network_name,vias):
	gpu_overhead=gpusim.get_network(network_name,vias)

def pipeline_aggressive(depth,reram, wires, gpu, batch_size):
	latency, energy = 0, 0 #ms, mj
	wire_latency, wire_energy=0,0
	gpu_latency,gpu_energy=0,0
	maxlatency=0
	for i in range(batch_size*(3*depth)):
		layer_latency = 0
		j= i%(2*depth+1)
		if j < depth:
			layer = reram[j]
			if layer['name']=='data': 
				continue
			layer_latency= layer['compute_latency'] if 'compute_latency' in layer else layer['latency']
			energy += layer['write_energy'] if 'write_energy' in layer else layer['energy']
			#energy += (layer["write_energy"]+layer['compute_energy']) if 'write_energy' in layer else layer['energy']
			maxlatency=max(maxlatency,layer_latency)
		elif j == depth:
			layer_latency +=maxlatency
		elif j > depth:
			idx = j - depth
			s = -idx
			layer = reram[s]
			#print("reram layer,",layer['name'])
			if layer['name']=='data': 
				continue
			layer_latency= layer['E_compute_latency'] if 'E_compute_latency' in layer else layer['BPlatency']
			#energy +=(layer["E_compute_energy"]+layer["E_write_energy"]) if 'write_energy' in layer else layer['energy']
			energy +=layer["E_compute_energy"] if 'write_energy' in layer else layer['energy']
		if i > depth:
			s = (i - depth-1)%depth
			print('wire s is ', s)
			delta = wires[-s-1]
			if s < depth-1:
				activation = wires[-s-2]
				print("wirelayer", delta['name'],activation['name'])
				layer_latency = max(delta['latency']+activation['latency'], layer_latency)
				wire_latency += delta['latency']+activation['latency']	
				wire_energy += delta['energy']+activation['energy']	
		if i >= depth+2:
			s = (i - depth-2)%(depth-1)
			#print("s is {}".format(s))
			gpu_layer = gpu[-s-1]
			layer_latency = max(gpu_layer['G_latency'] if 'G_latency' in gpu_layer else gpu_layer['BPlatency'], layer_latency)
			energy+= gpu_layer['G_energy'] if 'G_energy' in gpu_layer else gpu_layer['BPenergy']
		latency += layer_latency
		print("layer_latency {:.3f}".format(layer_latency))	
	if 'update_latency' in wires[-1].keys():
		latency += wires[-1]['update_latency']
		energy += wires[-1]['update_energy']
		wire_latency += wires[-1]['update_latency'] 	
		wire_energy += wires[-1]['update_energy']	
	print("layer_latency {:.3f},latency {:.3f}, energy {:.3f}".format(layer_latency,latency,energy))	
	
def pipeline_conservative(depth,reram, wires,gpu,batch_size):
	latency, energy = 0, 0 #ms, mj
	wire_latency, wire_energy=0,0
	gpu_latency,gpu_energy=0,0
	maxlatency=0
	for i in range(batch_size*(3*depth)):
		layer_latency, layer_energy = 0,0
		if i < depth:
			layer = reram[i]
			if layer['name']!='data': 
				#print('reram', layer['name'])
				layer_latency= layer['compute_latency'] if 'compute_latency' in layer else layer['latency']
				energy +=layer["write_energy"] if 'write_energy' in layer else layer['energy']
				maxlatency=max(maxlatency,layer_latency)
		if i >= depth:
			s = (i - depth)%depth
			#print("wire s {}, i {}".format(s,i))
			activation = wires[-s-1]
			#print('wire_layer', activation['name'])
			layer_latency = max(activation['latency'], layer_latency)
			energy += activation['energy']
			wire_latency += activation['latency']	
			wire_energy += activation['energy']	
		if i == depth+1:
			s = (i - depth-1)%(depth-1)
			gpu_layer = gpu[-s-1]
			#print('gpu_layer', gpu_layer['name'])
			layer_latency = max(gpu_layer['BPlatency'], layer_latency)
			energy +=gpu_layer['BPenergy']
		if i > depth+1:
			s = (i - depth-2)%(depth-1)
			gpu_layer = gpu[-s-1]
			#print('gpu_layer', gpu_layer['name'])
			layer_latency = max(gpu_layer['BPlatency'], layer_latency)
			energy +=gpu_layer['BPenergy']
		latency += layer_latency
		print("layer_latency {:.3f}".format(layer_latency))	
	if 'update_latency' in wires[-1].keys():
		layer_latency += wires[-1]['update_latency']
		energy += wires[-1]['update_energy']
		wire_latency += wires[-1]['update_latency'] 	
		wire_energy += wires[-1]['update_energy']	
	print("layer_latency {:.3f},latency {:.3f}, energy {:.3f}".format(layer_latency,latency,energy))	

def nopipeline_contrain(reram, datapath, gpu):
	wire_latency, wire_energy=0,0
	gpu_latency,gpu_energy=0,0
	#for layer in reram:
	#pass

def reram_batch(reram,train,bit_width):
	latency, energy = 0, 0 #ms, mj
	maxlatency=0
	degree = 1 
	for layer in reram:
		"""
		Load data into the pipeline
		"""
		if layer["name"].startswith("fc") or layer["name"].startswith("conv") or layer["name"].startswith("ip"):
			latency += max(layer["compute_latency"]/degree,layer["write_latency"])
			energy += layer["compute_energy"]*degree
			if train:
				energy +=layer["write_energy"]
				maxlatency=max(maxlatency,layer["compute_latency"],layer["write_latency"])
		elif layer["name"].startswith("pool") or layer["name"].startswith("relu"):
			latency += layer["latency"]
			energy += layer["energy"]
			if train:
				maxlatency=max(maxlatency,layer["latency"])
	
	if train==True:
		print("maxlatency is {:.3f} ".format(maxlatency))
		"""
		one cycle for loss
		"""
		latency +=maxlatency
		weight_rows, weight_columes =0,0
		for i  in range(len(reram)):
			j = -i-1
			array = reram[j]
			if array["name"].startswith("fc") or array["name"].startswith("conv") or array["name"].startswith("ip"):
				latency += max(array["compute_latency"]/degree,array["BPwrite_latency"])
				energy += (array["BPcompute_energy"]+array["BPwrite_energy"])*degree
			elif array["name"].startswith("pool") or array["name"].startswith("relu"):
				latency += array["BPlatency"]
				energy += array["BPenergy"]
			
			elif array["name"]=="update":
				latency += array["latency"]
				energy += array["energy"]	
				print("update latency {}, energy {}".format(array["latency"],array["energy"]))
			"""
			weight_rows += array["weight_rows"]
			weight_columes+=array["weight_columes"]
	"""		"""
	if batch_size 
	"""
	return latency,energy
				
def input():
	file_name =["","lenet","cifar10_full","alexnet.deploy","VGG_ILSVRC_16_layers_deploy"]
	network=[]
	wire_overhead = []
	reram_overhead=[]
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--bit_width',dest='BW',type=int, help='Training input batch size', default=8)
	parser.add_argument('-fp', '--inference ',dest='fp',type=bool, help='Number of Degrees', default=False)
	args = parser.parse_args()
#	os.chdir(args.workdir)
#    	outdir = 'output/'
#    	if not os.path.exists(outdir):
#        	os.makedirs(outdir)
#    	outfile = outdir + ntpath.basename(arg.infile)
	##
	wl,cl=args.wl,args.cl
	for i in range(0,len(file_name)):
		egdir = 'examples/'
		model = egdir + file_name[i]+".prototxt"
		if not os.path.exists(model):
			print("{} not exsited!".format(model))
			continue	
		print("model {}".format(file_name[i]))
		parse_input(network,model)
		get_reram(reram_overhead, network)
		wire_overhead = get_wire_overhead(network)

def test_agg():
	network=[]
	train=True
	bit_width = 32
	bit_per_cell = 4
	cells = bit_width/bit_per_cell
	vias = 1
	file_name = "examples/lenet.prototxt"
	parse_input(network,file_name)
	reram = get_reram(network,train,cells)
	gpu=gpusim.get_network('lenet',vias)
	wires = get_wire_overhead(network,vias)
	print("Conservative:")
	pipeline_conservative(len(network),reram,wires,gpu,1)	
	print("Aggressive:")
	pipeline_aggressive(len(network),reram,wires,gpu,1)	


if __name__ == '__main__':
	test_agg()
		
"""
	reram = [{'name':'conv', 'compute_latency': 3, 'write_latency':4,'compute_energy':3, "E_compute_latency":5,"E_compute_energy":2.5},{'name':'pool','BPlatency':2.1, 'BPenergy':2,'latency':2, 'energy':2}]
	wires= [{'name':'conv','latency':1,'energy': 0.3},{'name':'pool','energy':0.5,'latency':1}]
	gpu	 = [{'name':'conv','latency':12,'energy':0.1},{'name':'pool','latency':4,'energy':0.2}]
	depth = len(reram)
	pipeline_aggressive(depth,reram,wires,gpu,1)	
	network=[]
	reram_overhead=[]
	train = False
	#file_name = "examples/cifar10_full.prototxt"
	file_name = "examples/lenet.prototxt"
	parse_input(network,file_name)
	reram_overhead=get_reram(network,train,cells)
	for layer in reram_overhead:
		print(layer)
	overhead=reram_batch(reram_overhead,train,1)
	print("latency(ns), energy(nJ)")
	print("{:.3f} {:.3f}".format(overhead[0]*1e6,overhead[1]*1e6))
	"""
	
"""
def train(model_name, epoches, batch_size):
	network = []
	top = ['name', 'in_num', 'kernel_size','out_num','out_size']
	tot_tsv_latency, tot_tsv_energy = 0,0
	tot_m3d_latency, tot_m3d_energy = 0,0
	design=[0,1] # 0:consevation, 1: aggressive
	interconnect=[0,1,2] #0:2D, 1: TSV, 2: M3D
	for j in range(epoches):
	# interconnect = TSV
	# Conservative design
	tsv_latency, tsv_energy=pipeline1(network,array,1) if design==0 else pipeline2(network,array,1)
	tot_tsv_latency += tsv_latency
	tot_tsv_energy  += tsv_energy

	m3d_latency, m3d_energy=pipeline1(network,array,2) if design==0 else pipeline2(network,array,2)
	tot_m3d_latency += m3d_latency
	tot_m3d_energy  += m3d_energy

	if design==0:
		print("Conservative\ntsv overhead ms, mJ, m3d overheda ns, nJ")
		print(" %.3f, %3f, %.3f, %.3f"%, tot_tsv_latency/1000000,tot_tsv_energy/1000000,m3d_latency/1000000,m3d_energy/1000000)
	if design==1:
		print("Aggressive\ntsv overhead ms, mJ, m3d overheda ns, nJ")
		print(" %.3f, %3f, %.3f, %.3f"%, tot_tsv_latency/1000000,tot_tsv_energy/1000000,m3d_latency/1000000,m3d_energy/1000000)

	

"""	
	
	
