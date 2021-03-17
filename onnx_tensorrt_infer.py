import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import tensorrt as trt


model_rpn = onnx.load("rpn_slim.onnx")
engine_rpn = backend.prepare(model_rpn, device='CUDA:0')
input_data_rpn = np.ones((1, 64, 496, 432)).astype(np.float32)
output_data_rpn = engine_rpn.run(input_data_rpn)
print(output_data_rpn)


# model_pfn = onnx.load("pfn_slim.onnx")
# engine_pfn = backend.prepare(model_pfn, device='CUDA:0', max_batch_size=1)
#
# pillar_x = np.ones((40000, 32, 1)).astype(np.float32)
# pillar_y = np.ones((40000, 32, 1)).astype(np.float32)
# pillar_z = np.ones((40000, 32, 1)).astype(np.float32)
# pillar_i = np.ones((40000, 32, 1)).astype(np.float32)
# x_sub_shaped = np.ones((40000, 32, 1)).astype(np.float32)
# y_sub_shaped = np.ones((40000, 32, 1)).astype(np.float32)
# pfn_input_num_points = np.ones(40000).astype(np.float32)
# pfn_input_mask = np.ones((40000, 32, 1)).astype(np.float32)
# output_data_pfn = engine_pfn.run([pillar_x, pillar_y, pillar_z, pillar_i, pfn_input_num_points, x_sub_shaped, y_sub_shaped, pfn_input_mask])
# print(output_data_pfn)



