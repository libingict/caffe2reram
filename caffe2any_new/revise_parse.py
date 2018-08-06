import sys
import argparse
import caffe_pb2 as caffe
from google.protobuf import text_format
import math
import numpy as np
import ntpath
import gpusim
import os

top = ['name', 'in_num', 'kernel_size','out_num','out_size']
CELL_TYPE = 2
BIT_WIDTH = 16
N = math.ceil(BIT_WIDTH/CELL_TYPE)
"""
wires_overhead
inputs: data, without bit-width, if float, then input *32; else input*16
vias: vias's type, 0:2D, 1:TSV, 3:M3D
return latency(ns), energy(nJ)
"""

def wires_overhead(network,vias,bit_width=1):
	"""
	The interconnect data are from "Architecting Large-Scale SRAM Arrays with Monolithic 3D Integration, ISLPED17".Fig 8, 64Mb Data Array
	dynamic enery
	"""
	DDR=[80,1.02]        #GB/s, nJ
	TSV = [355, 0.39]    #GB/s, nJ
	MIV = [5800,0.25]   #GB/s, nJ
	tot_weights=0
	overhead=[]
	if vias==0:
		rates=DDR[0]
		energy=DDR[1]
	elif vias==1:
		rates,energy=TSV[0],TSV[1]
	else:
		rates,energy=MIV[0],MIV[1]
	for layer in network:
		latency,energies,weights=0,0,0	
		data_volume = layer["out_size"]**2*layer["out_num"]
		latency = float(data_volume*bit_width/rates)*1e-6  #ms
		energies = float(energy*bit_width*data_volume)*1e-6  #mJ
		overhead.append({"name":layer["name"],
				"latency":latency,
				"energy":energies})
	
		if layer["name"][0:4]=="conv":
			weights = layer["out_num"]*layer["kernel_size"]**2
		#get (out_num) output neurons from the weight array 	
		elif layer["name"][0:2]=="fc" or layer["name"][0:2]=="ip":
			weights = layer["kernel_size"]* layer["out_num"]
		tot_weights +=weights
	
	latency = float(tot_weights*bit_width/rates)*1e-6  #ms
	energies = float(energy*bit_width*tot_weights)*1e-6  #mJ
	overhead[-1]["update_latency"]=latency
	overhead[-1]["update_energy"]=energies
		
	return overhead
"""

the row number is kernel_size*kernel_size*input_number;

the column number is output_number * N (N = bit-width/cell-bits)
degree:  

cycle = outputsize^2*N / degree 
col: horizontally collect, outnumber
row: vertically sum * degree
get (out_num) output neurons from the weight array 	

"""
def get_weightarray(rows,columes,wl=128, cl=128):
	# degree, numofwordline, number of columnline
	#print("subarray size is %2dx%d, Parallelism degree %2d"%(wl,cl, degree))
	row_part = math.ceil(rows/wl) #vertically sum
	col_part = math.ceil(columes/cl)
	#print("FF subarray num is %2d(inpart)x %2d(outpart)"% (row_part,col_part))
	return row_part, col_part
"""
"""

def get_row_colume(layer,nextlayer=None):
	rows, columes = 0,0 
	if layer["name"][0:4]=="conv":
		rows = layer["in_num"]*layer["kernel_size"]**2
		columes = layer["out_num"]
	#get (out_num) output neurons from the weight array 	
	#degree: the number of output neurons one time 
	elif layer["name"][0:2]=="fc" or layer["name"][0:2]=="ip":
		rows = layer["kernel_size"]
		columes = layer["out_num"]
	elif layer["name"][0:4]=='data' or layer["name"][0:4]=='pool':
		rows = layer["kernel_size"]
		columes = layer["kernel_size"]
	return rows, columes

def get_buffer_array(layer):
	rows = layer["out_size"]**2
	columes = layer["out_num"]
	##overhead im2col_overhead()
	#get (out_num) output neurons from the weight array 	
	#degree: the number of output neurons one time 
	return rows, columes
	
"""

top = ['name', 'in_num','kernel_size','out_num','out_size']

[data, input_channel,input_height,input_channel,input_width]
[conv, in_num, kernel_size, out_num, out_size]
[fc, in_num, kernel_size, out_num, 1]

"""
def tile_design(layers):
	xbar=[]
	for i in range(len(layers)):
		rows,columes=0,0
		layer = layers[i]
		rows, columes=get_row_colume(layer)
		brows, bcolumes=get_buffer_array(layer)
		#print(layer, rows, columes)
		array_layer = {"name":layer["name"],
				"weight_rows":rows,
				"weight_columes":columes,
				"buffer_rows":brows,
				"buffer_columes":bcolumes}
		xbar.append(array_layer)      	
	return xbar

"""
xbar: the array 
Ncells: if 4, then 16fixed, if 8, then 32fixed
"""
def cost(xbar,train,Ncells):
	read_energy, write_energy = 1.08,3.91 #pJ, nJ
	read_latency, write_latency = 29.31,50.88 #ns, cell * cell#
	tot_weights_rows, tot_weights_columes = 0,0
	maxlantency = 0
	for i in range(len(xbar)):
		tot_write_latency,tot_write_energy = 0,0
		array = xbar[i]
		if array["name"]=="data":
			continue
		if array["name"].startswith("relu"):
			array["latency"] = write_latency*1e-6
			array["energy"] = 0.23*array["latency"]*1e-9 #mW * ns, 1e-9mJ
		elif array["name"].startswith("pool"):
			array["latency"]= array["buffer_columes"] * write_latency * 1e-6 
			array["energy"] = array["buffer_columes"] * array["buffer_rows"] * write_energy *Ncells*1e-6
		else:
			compute_latency =  read_latency * array["buffer_rows"] 
			compute_energy = array["weight_columes"] * array["weight_rows"] * Ncells * read_energy * array["buffer_rows"]

			if i < len(xbar)-1:
				if xbar[i+1]["name"].startswith("pool"):
					written_row = min(xbar[i+1]["weight_rows"],array["buffer_rows"])
				else:
					written_row = array["buffer_rows"]
			else:
				written_row = array["buffer_rows"]
			#print("FF layer {} written_row {} colume {}".format(array["name"],written_row, array["buffer_columes"]))
			tot_write_energy = array["buffer_columes"] * written_row * write_energy * Ncells
			tot_write_latency = written_row * write_latency 
			#print("FF layer write_energy {}".format(tot_write_energy))
			array["compute_latency"]= compute_latency*1e-6
			array["write_latency"]= tot_write_latency*1e-6 #ms
			array["compute_energy"] = compute_energy*1e-9 
			array["write_energy"] = tot_write_energy *1e-6 #mj
	if train:
		for i in range(len(xbar)):
			j = -i-1
			array = xbar[j]	
			G_write_latency,G_write_energy = 0,0
			E_write_latency,E_write_energy = 0,0
			E_write_latency,E_write_energy = 0,0
			if array["name"]=="data":
				break
			if array["name"].startswith("relu"):
				array["BPlatency"] = write_latency*1e-6
				array["BPenergy"] = 0.23*array["latency"]*1e-9 #mW * ns, 1e-9mJ
			elif array["name"].startswith("pool"):
				array["BPlatency"]= array["buffer_columes"] * write_latency* 1e-6
				array["BPenergy"] = array["buffer_columes"] * array["buffer_rows"] * write_energy *Ncells*1e-6
			else:
				tot_weights_rows +=array["weight_rows"]
				tot_weights_columes += array["weight_columes"]
				#dx, transpose weights array  [weight_rows][buffer_rows]
				compute_latency = array["buffer_rows"] *  read_latency
				compute_energy = array["weight_columes"]* array["weight_rows"] * Ncells * read_energy * array["buffer_rows"]
				array["E_compute_latency"]= compute_latency*1e-6
				array["E_compute_energy"] = compute_energy*1e-9 
				#dw, 
				compute_latency =  read_latency * array["weight_columes"]
				compute_energy = array["buffer_rows"]* array["weight_rows"] * Ncells * read_energy * array["weight_columes"]
				array["G_compute_latency"]= compute_latency*1e-6
				array["G_compute_energy"] = compute_energy*1e-9 
				#write dx
				if j < 0:
					if xbar[j+1]["name"].startswith("pool"):
						written_row = min(xbar[j+1]["weight_rows"],array["weight_rows"])
					else:
						written_row = array["weight_rows"]
					#print("BP layer {} written_row {} colume {}".format(array["name"],written_row, array["buffer_rows"]))
					E_write_energy = array["buffer_rows"]*written_row * write_energy
					E_write_latency = written_row * write_latency * Ncells
				# write dw 
					G_write_energy = array["weight_rows"]*array["weight_columes"] * write_energy *Ncells
					G_write_latency = array["weight_rows"] * write_latency 

				array["E_write_latency"] = E_write_latency*1e-6 #ms
				array["E_write_energy"] = E_write_energy *1e-6 #mj
				array["G_write_latency"] = G_write_latency*1e-6 #ms
				array["G_write_energy"] = G_write_energy *1e-6 #mj
				array["BPcompute_latency"]= array["E_compute_latency"]+array["G_compute_latency"]
				array["BPcompute_energy"] = array["E_compute_latency"] + array["G_compute_energy"]
				array["BPwrite_energy"] = array["E_write_energy"] + array["G_write_energy"]  #mj
				array["BPwrite_latency"] = array["E_write_latency"] + array["G_write_latency"] #ms
				
		##Update all weights arrays
		#print("tot_weights_columes {}, tot_weights_rows {}".format(tot_weights_columes, tot_weights_rows))
		update_latency = tot_weights_columes * write_latency *1e-6
		update_energy = tot_weights_columes * Ncells * tot_weights_rows * write_energy *1e-6
			
	
		xbar[-1]["update_latency"]= update_latency
		xbar[-1]["update_energy"] = update_energy
		print("update_energy {}, update_latency {}".format(update_energy,update_latency))
	#return xbar 

"""
Per Layer: Parallelism Degree; Per Layer's cycle(conv+activation; pooling); Per Layer's energy (nat)
input: layer: each layer of model, 
input: array_design: rows's number, columes's number, and buffer size
return: latency, and energy 
"""
	
def Comparison_overhead(xbar,nettype,Ncells,vias,batch_size):
	wires_latency, wires_energy=0,0
	batch_latency, batch_energy=0,0
	tot_weights=0
	tot_activations=0
	for l in range(1,len(network)): #without the data layer
		activations=0
		if network[-l]["name"][0:4]=="pool" or (network[-l]["name"][0:4]=="conv" and network[-l+1]["name"]!="pool"):
			activations=network[-l]['out_num']*network[-l]['out_size']*network[-l]['out_size']  #data without bit-width
		if network[-l]["name"][0:4]!="pool" :
			tot_weights +=network[-l]['out_num']*network[-l]['kernel_size']*network[-l]['kernel_size']
		tot_activations +=activations
	
	inputs = network[0]['in_num']*network[0]['kernel_size']*network[0]['kernel_size']
	latency,energy =wires_overhead(tot_activations+inputs,vias,Ncells)

	wires_latency += latency
	wires_energy += energy 

	latency,energy = wires_overhead(tot_weights,vias,Ncells)
	wires_latency +=latency
	wires_energy +=energy
	tot_wire_latency=wires_latency*(batch_size-1)
	tot_wire_energy=wires_energy*(batch_size-1)
	print("wire(ns nj), %.2f , %.2f"% (tot_wire_latency, tot_wire_energy))
	gpu_energy,gpu_latency=gpusim.gpu_eval(nettype,vias,design=0)
	print("gpu(ns nj), %.2f , %.2f"% (gpu_latency, gpu_energy))
	reram_latency, reram_energy = cost(xbar,Ncells,False)
	batch_latency = reram_latency*batch_size+tot_wire_latency+gpu_latency
	batch_energy = reram_energy*batch_size+gpu_energy+tot_wire_energy
	return batch_latency, batch_energy

	#ReRAM compute the delta
"""
model:  neural model, list
memory: memory model, list, [array number, buffer size]
interconnect: connections type, int, 0:2D, 1: TSV, 2: M3D
return: the longest latency for ff, interconnection and gpu

""" 
def MIV_overhead(Ncells,batch_size,design):
	tot_weights,tot_activations=0,0
	wires_energy,wires_latency,latency,energy=0,0,0,0
	for l in range(2,len(network)): #without the last and data layer
		activations=0
		if network[-l]["name"][0:4]=="pool" or (network[-l]["name"][0:4]=="conv" and network[-l+1]["name"]!="pool"):
			activations=network[-l]['out_num']*network[-l]['out_size']*network[-l]['out_size']  #data without bit-width
		if network[-l]["name"][0:4]!="pool":
			tot_weights +=network[-l]['out_num']*network[-l]['kernel_size']*network[-l]['kernel_size']
		tot_activations = tot_activations+activations if design==0 else tot_activations+2*activations 

	inputs = network[0]['in_num']*network[0]['kernel_size']*network[0]['kernel_size']
	loss = network[-1]['out_num']*network[-1]['out_size']*network[-1]['out_size']
	if design==0:
		latency,energy =wires_overhead(tot_activations+inputs+loss,2,Ncells)
	else:
		latency,energy =wires_overhead(tot_activations+inputs,2,Ncells)
	wires_latency += latency
	wires_energy += energy 
	latency,energy = wires_overhead(tot_weights,2,Ncells)
	if design ==0 :
		print("Conservative wire(ns nj), %.2f , %.2f"% (wires_latency*(batch_size-1)+latency, wires_energy*(batch_size-1)+energy))
	else:
		print("Aggressive wire(ns nj), %.2f , %.2f"% (wires_latency*(batch_size-1)+latency, wires_energy*(batch_size-1)+energy))
	return wires_latency,wires_energy

#call gpu's overhead, two conv cmputation, that's train-ff;
"""
pipeline
design = 0, conservative
design = 1, aggressive
"""	
def pipeline(xbar, nettype, Ncells, batch_size,design=0):
	reram_latency, reram_energy = 0,0
	depth = len(xbar) 
	gpu_latency, gpu_energy = 0,0
	batch_latency, batch_energy = 0,0
	wire_latency,wire_energy = MIV_overhead(Ncells,batch_size,design)
	gpu_energy,gpu_latency=gpusim.gpu_eval(nettype,2,design)
	if design == 0:
		print("Conservative gpu(ns nj), %.2f , %.2f"% (gpu_latency, gpu_energy))
	else:
		print("Aggressive gpu(ns nj), %.2f , %.2f"% (gpu_latency, gpu_energy))
	if design == 0:
		reram_latency, reram_energy = cost(xbar,Ncells,False)
	else:
		ff_latency,ff_energy = cost(xbar,Ncells,False)	
		bp_latency,bp_energy = cost(xbar,Ncells,True)
		reram_latency = ff_latency + bp_latency
		reram_energy =ff_energy+bp_energy
		print("2DReRAM latency(ns) energy(nj), %.3f, %.3f"% (reram_latency+(batch_size-1)*(reram_latency/depth),reram_energy*batch_size))

	batch_latency = max(reram_latency*batch_size,wire_latency,gpu_latency)
	batch_energy = reram_energy*batch_size+gpu_energy+wire_energy
	#print("Design latency(ns) energy(nj), %f, %f"% (batch_latency,batch_energy))
	return batch_latency, batch_energy

#return max_subarray
def addinnetwork(network,value):
	top = ['name', 'in_num','kernel_size','out_num','out_size']
	new_layer={}
	for i in range(len(top)):
		new_layer[top[i]] = value[i]
	network.append(new_layer)

"""

top = ['name', 'in_num','kernel_size','out_num','out_size']

[data, input_channel,input_height,input_channel,input_width]
[conv, in_num, kernel_size, out_num, out_size]
[fc, in_num, kernel_size, out_num, 1]

"""

def parse_input(network,file_name):
	top = ['name', 'in_num', 'kernel_size','out_num','out_size']
	net = caffe.NetParameter()
	try:
		f = open(file_name, "rb")
		text_format.Parse(f.read(), net)
		f.close()
	except IOError:
		exit("Could not open file " + sys.argv[1])
	
	for layer in net.layer:
		if layer.name == 'data':
#			print("layer.name ", layer.name)
			if layer.input_param is not None:
				if layer.input_param.shape is not None and len(layer.input_param.shape)>0:
					if len(layer.input_param.shape[0].dim) == 4:
						batch,input_channel,input_height,input_width = layer.input_param.shape[0].dim
						addinnetwork(network, ['data',input_channel,input_height,input_channel, input_width])
		elif layer.name[0:4]=='conv':
#			print("layer.name ", layer.name)
			if len(network) > 0:
				prelayer = network[-1]
				in_num=network[-1]['out_num']
				prelayer_size = network[-1]['out_size']
				if layer.convolution_param is not None:
					if layer.convolution_param.num_output is not None:
						#print("layer: ", layer.name, "num_output ", layer.convolution_param.num_output)
						out_num=layer.convolution_param.num_output
					## get its bottom and get the bottom's num_output
						if layer.convolution_param.kernel_size is not None:
						## the kernel's data volume is 
							kernel_size = layer.convolution_param.kernel_size	
						else:
							kernel_size=0
						if layer.convolution_param.pad is not None:
							pad = layer.convolution_param.pad
						else:
							pad = 0
						if layer.convolution_param.stride is not None:
							stride = layer.convolution_param.stride
						else:
							stride = 1
						if layer.convolution_param.group is not None:
							group = layer.convolution_param.group
						else:
							group = 1
						
						out_num= layer.convolution_param.num_output * group
						out_size = math.ceil((prelayer_size-kernel_size+2*pad)/stride)+1						
						#print("layer, ", layer.name, " in_num: ", in_num, " kernel_size: ", kernel_size, " out_num: ", out_num, " out_size: ", out_size)
						addinnetwork(network,[layer.name, in_num, kernel_size, out_num, out_size])
		elif layer.name[0:4] == 'relu':
			prelayer = network[-1]
			in_num=network[-1]['out_num']
			prelayer_size = network[-1]['out_size']
#			print("layer.name ", layer.name)
			addinnetwork(network,[layer.name,in_num,0,in_num, prelayer_size])
		elif layer.name[0:4]=='pool':
			if len(network) > 0:
				prelayer = network[-1]
				in_num=network[-1]['out_num']
				prelayer_size = network[-1]['out_size']
				if layer.pooling_param is not None:
					## get its bottom and get the bottom's num_output
						if layer.pooling_param.kernel_size is not None:
						## the kernel's data volume is 
							kernel_size = layer.pooling_param.kernel_size	
						else:
							kernel_size=2
						if layer.pooling_param.pad is not None:
							pad = layer.pooling_param.pad
						else:
							pad = 0
						if layer.pooling_param.stride is not None:
							stride = layer.pooling_param.stride
						else:
							stride = 1
						out_num=in_num 
						out_size = math.ceil((prelayer_size-kernel_size+2*pad)/stride)+1						
						#print("layer, ", layer.name, " in_num: ", in_num, " kernel_size: ", kernel_size, " out_num: ", out_num, " out_size: ", out_size)
						addinnetwork(network, [layer.name, in_num, kernel_size, out_num, out_size])
		elif layer.name[0:2]=='fc' or layer.name[0:2]=='ip':
			if len(network) > 0:
				prelayer = network[-1]
				in_num=1
				prelayer_size=network[-1]['out_num']*network[-1]['out_size']**2
				#prelayer_size = network[-1]['out_size']
				if layer.inner_product_param is not None:
					## get its bottom and get the bottom's num_output
						if layer.inner_product_param.num_output is not None:
							out_num=layer.inner_product_param.num_output
						else:
							out_num=1
						out_size = 1						
						addinnetwork(network, [layer.name, in_num, prelayer_size, out_num, out_size])
def print_array(net):
	for layer in net:
		for keys, values in layer.items():
			print("{} {}".format(keys, values))
if __name__ == '__main__':
#	main()
#	for layers in network:
#	       for key in layers.keys():
	file_name = "examples/lenet.prototxt"
	#file_name = "examples/cifar10_full.prototxt"
	network=[]
	parse_input(network,file_name)
	overhead=[]
	wires_overhead(network,overhead,0,1)
	print(overhead)
	xbar=tile_design(network)#, degree=1, Ncells=1, batch_size=1,wl=128, cl=128)
	cost(xbar,True,8)
	"""
	for layer in xbar:
		print(layer)
		if layer["name"].startswith("fc") or layer["name"].startswith("conv") or layer["name"].startswith("ip"):
			print(layer["name"])
			for keys,values in layer.items():
				print("{}, {}".format(keys, values))
	"""
