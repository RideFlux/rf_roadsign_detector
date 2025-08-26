import hydra
from omegaconf import DictConfig, OmegaConf
from onnxsim import simplify
import torch
import torch.nn as nn
import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from utils.sum_resolver import SumResolver


OmegaConf.register_new_resolver("sum", SumResolver, replace=True)

def generate_default_onnx(detector, 
                          BATCH_SIZE, MAX_N_VOXELS, MAX_POINTS_PER_VOXEL, N_CHANNEL_POINT,
                          ):
    #====================================================================================================#
    # pytorch 코드를 읽어 초기 onnx를 만들어내는 과정
    #====================================================================================================#
    dummy_voxels = torch.zeros(
        (MAX_N_VOXELS, MAX_POINTS_PER_VOXEL, N_CHANNEL_POINT),
        dtype=torch.float32,
        device='cuda:0')

    dummy_voxel_idxs = torch.zeros(
        (MAX_N_VOXELS, N_CHANNEL_POINT),
        dtype=torch.int64,
        device='cuda:0')

    dummy_voxel_num = torch.zeros(
        (1,),
        dtype=torch.int32,
        device='cuda:0')

    dummy_input = dict()
    dummy_input['voxels'] = dummy_voxels
    dummy_input['voxel_num_points'] = dummy_voxel_num
    dummy_input['voxel_coords'] = dummy_voxel_idxs
    dummy_input['batch_size'] = BATCH_SIZE

    tmp_onnx_file_name = 'tmp.onnx'
    
    torch.onnx.export(detector,       
        (dummy_input, {}),               
            tmp_onnx_file_name,  
        export_params=True,        
        do_constant_folding=True,  
        keep_initializers_as_inputs=True,
        input_names = ['voxels', 'voxel_num', 'voxel_idxs'],   
        output_names = ['final_boxes', 'final_labels', 'final_scores'], 
        )

    onnx_raw = onnx.load(tmp_onnx_file_name)
    os.remove(tmp_onnx_file_name)

    onnx_simp, check = simplify(onnx_raw)
    assert check, "Simplified ONNX model could not be validated"
    
    return onnx_simp

def modify_onnx(onnx_simp,
                BATCH_SIZE, MAX_N_POINTS, N_CHANNEL_POINT, MAX_N_VOXELS, MAX_POINTS_PER_VOXEL,
                POINT_CLOUD_RANGE, N_CHANNEL_PILLAR, VOXEL_SIZE):
    
    #====================================================================================================#
    # 생성된 초기 onnx를 변형해 plugin을 붙이고 불필요한 노드를 없애는 과정
    #====================================================================================================#
    
    graph = gs.import_onnx(onnx_simp)
    tmap = graph.tensors()

    # 새롭게 input을 정의하기 위해 만들음. 기존에 voxel을 input으로 받는 onnx를 생성했지만, plugin을 연결해서 point를 input으로 받도록 변경시켜줄 것
    points = gs.Variable(name="points", dtype=np.float32, shape=(BATCH_SIZE, MAX_N_POINTS, N_CHANNEL_POINT))
    num_points = gs.Variable(name="num_points", dtype=np.int32, shape=(BATCH_SIZE, ))

    # voxel generator에 넣어 줄 output variable을 만들어 줌
    voxel_inputs = gs.Variable(name="voxel_inputs", dtype=np.float32, shape=(1, MAX_N_VOXELS, MAX_POINTS_PER_VOXEL, 10))
    voxel_coords = gs.Variable(name="voxel_coords", dtype=np.int32, shape=(1, MAX_N_VOXELS, 4))
    num_pillar = gs.Variable(name="num_pillar", dtype=np.int32, shape=(1,))

    # plugin을 통해서 voxel을 만들어 줌. config에 정의되어 있는 값을 가져와 사용함
    voxel_gen = gs.Node(
        name="voxelGen",
        op="VoxelGeneratorPlugin", 
        attrs={
            "max_num_points_per_voxel": MAX_POINTS_PER_VOXEL,
            "max_voxels": MAX_N_VOXELS,
            "point_cloud_range": POINT_CLOUD_RANGE,
            "voxel_feature_num": N_CHANNEL_PILLAR,
            "voxel_size": VOXEL_SIZE,
            }, 
        inputs=[points, num_points], 
        outputs=[voxel_inputs, voxel_coords, num_pillar]
        )

    graph.nodes.append(voxel_gen) # 만들어 낸 voxel generator를 onnx graph에 추가하는 과정

    '''
    pointpillar pytorch 코드에서 만들어 낸 onnx는 f_cluster, f_center 계산 등 복잡한 연산이 함께 들어가있음
    voxel generator의 output은 10 channel로 기존에 pytorch에서 계산하던 과정이 다 포함되어있으므로 위의 과정을 끊어줄 필요가 있음
    따라서 새로운 node를 만들고 필요한 op만 찾아서 연결시켜주면 simple한 onnx를 만들어 낼 수 있음
    아래 단계에서는 pillar_vfe.py 에 정의 되어있는 코드를 수행시키는 onnx를 만들어 냄
    '''
    reshape_0 = gs.Node(name="reshape_0", op = "Reshape")
    reshape_0.inputs.append(voxel_inputs)
    reshape_0_shape = gs.Constant(name="reshape_0_shape", values = np.array([MAX_N_VOXELS * MAX_POINTS_PER_VOXEL, 10], dtype=np.int64))
    reshape_0.inputs.append(reshape_0_shape)
    reshape_0_out = gs.Variable(name="reshape_0_out", shape = [MAX_N_VOXELS * MAX_POINTS_PER_VOXEL, 10], dtype=np.float32)
    reshape_0.outputs.append(reshape_0_out) 
    graph.nodes.append(reshape_0)
    
    #====================================================================================================#

    matmul_op_0 = [node for node in graph.nodes if node.op == "MatMul"][0]
    matmul_op_0.inputs[0] = reshape_0_out
    matmul_op_0_out = gs.Variable(name="matmul_op_0_out", shape = [MAX_N_VOXELS * MAX_POINTS_PER_VOXEL, 64], dtype=np.float32)
    matmul_op_0.outputs[0] = matmul_op_0_out

    bn_op_0 = [node for node in graph.nodes if node.op == "BatchNormalization"][0]
    bn_op_0.inputs[0] = matmul_op_0_out
    bn_op_0_out = gs.Variable(name="bn_op_0_out", shape = [MAX_N_VOXELS * MAX_POINTS_PER_VOXEL, 64], dtype=np.float32)
    bn_op_0.outputs[0] = bn_op_0_out

    relu_op_0 = [node for node in graph.nodes if node.op == "Relu"][0]
    relu_op_0.inputs[0] = bn_op_0_out
    relu_op_0_out = gs.Variable(name="relu_op_0_out", shape = [MAX_N_VOXELS * MAX_POINTS_PER_VOXEL, 64], dtype=np.float32)
    relu_op_0.outputs[0] = relu_op_0_out
    #====================================================================================================#

    reshape_1 = gs.Node(name="reshape_1", op = "Reshape")
    reshape_1.inputs.append(relu_op_0_out)
    reshape_1_shape = gs.Constant(name="reshape_1_shape", values = np.array([MAX_N_VOXELS, MAX_POINTS_PER_VOXEL, 64], dtype=np.int64))
    reshape_1.inputs.append(reshape_1_shape)
    reshape_1_out = gs.Variable(name="reshape_1_out", shape = [MAX_N_VOXELS, MAX_POINTS_PER_VOXEL, 64], dtype=np.float32)
    reshape_1.outputs.append(reshape_1_out)
    graph.nodes.append(reshape_1)

    # voxel 내 point를 maxpooling 하여 voxel당 하나의 feature를 만들어 냄
    #====================================================================================================#
    reducemax_op_0 = [node for node in graph.nodes if node.op == "ReduceMax"][0]
    reducemax_op_0.inputs[0] = reshape_1_out
    reducemax_op_0.attrs['keepdims'] = [0]
    reducemax_op_out = gs.Variable(name="reducemax_0_op_out", shape = [MAX_N_VOXELS, 64], dtype=np.float32)
    reducemax_op_0.outputs[0] = reducemax_op_out
    #====================================================================================================#

    reshape_2 = gs.Node(name="reshape_2", op = "Reshape")
    reshape_2.inputs.append(reducemax_op_out)
    reshape_2_shape = gs.Constant(name="reshape_2_shape", values = np.array([1, MAX_N_VOXELS, 64], dtype=np.int64))
    reshape_2.inputs.append(reshape_2_shape)
    reshape_2_out = gs.Variable(name="voxels", shape = [1, MAX_N_VOXELS, 64], dtype=np.float32)
    reshape_2.outputs.append(reshape_2_out)
    graph.nodes.append(reshape_2)

    '''
        pointpillar_scatter.py에 해당하는 처리를 PillarScatterPlugin이 담당하는데 이것과 input & output을 엮어주기 위한 과정
        input:
            1) 위에서 MLP, maxpooling을 통과하여 만들어진 voxel feature
            2) voxel generator가 만들어 낸 voxel coordinate
            3) voxel generator가 만들어 낸 num_pillar
        output:
            2D conv라서 이를 연결해주기 위한 코드임

        이 부분이 plugin으로 대체되는 이유는 추측하기로
        1) onnx 단에서 variable인 유효한 voxel수를 나타내는 num_pillar를 처리할 수 없음. 그렇다면 MAX 사이즈에 대해서 다 scatter해 주어야 하는데,
        그렇게 되면 (0,0,0,0) coordinate값이 오염됨. 이게 큰 문제는 아닌 케이스가 많을것 같지만 어찌됐든 비효율이긴 함 또한 MAX 사이즈에 대해 전체 진행하면 약간의 비효율 발생
        2) pytorch에서 만들어 낸 onnx를 그대로 사용하면 node가 다소 지저분하기도 하고 이 부분 plugin으로 쓰지 않을때 대비 onnx 전체 용량이 3배이상임 (20mb -> 68mb).
        용량이 절대적으로 중요한 것은 아니지만 문제의 소지가 있음
    '''
    #====================================================================================================#
    @gs.Graph.register()
    def replace_with_clip(self, inputs, outputs, dense_shape):
        for inp in inputs:
            inp.outputs.clear()

        for out in outputs:
            out.inputs.clear()

        op_attrs = dict()
        op_attrs["dense_shape"] = dense_shape

        return self.layer(name="pillarScatter", op="PillarScatterPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

    conv_op = [node for node in graph.nodes if node.op == "Conv"][0]
    dense_shape = conv_op.inputs[0].shape[-2:]
    graph.replace_with_clip([reshape_2.outputs[0], voxel_coords, num_pillar], [conv_op.inputs[0]], dense_shape)
    #====================================================================================================#

    # 새롭게 만들어진 onnx에 대해 input과 output을 정의 함
    graph.inputs = [points, num_points]
    graph.outputs = [tmap['final_boxes'], tmap['final_labels'], tmap['final_scores']]
    graph.cleanup().toposort()
    onnx_final = gs.export_onnx(graph)
    onnx.save(onnx_final, "final.onnx")
    
    '''
    위 과정대로 하면 export용 pytorch 파일을 따로 만들지 않고 onnx단에서 처리 가능함
    단 get_rotated_boxes는 c++에서 연산해야 함
    '''
    
    return onnx_final

@hydra.main(version_base=None, config_path="configs", config_name="export.yaml")
def main(cfg: DictConfig):
    detector : nn.Module = hydra.utils.instantiate(cfg.model.roadsign_detector.module)
    checkpoint = torch.load(cfg.ckpt_path, weights_only=False)

    model_weights = checkpoint['state_dict']

    detector.load_state_dict(model_weights)
    detector.to("cuda:0")
    detector.eval()

    #====================================================================================================#
    BATCH_SIZE = 1
    N_CHANNEL_POINT = 4 # (x, y, z I)
    N_CHANNEL_PILLAR = 10 # (x, y, z, I, cluster_x, cluster_y, cluster_z, center_x, center_y, center_z)

    MAX_N_VOXELS = cfg.model.roadsign_detector.max_voxels[0]
    MAX_N_POINTS = cfg.model.roadsign_detector.max_points
    MAX_POINTS_PER_VOXEL = cfg.model.roadsign_detector.max_num_points
    VOXEL_SIZE = cfg.model.roadsign_detector.voxel_size
    POINT_CLOUD_RANGE = cfg.model.roadsign_detector.point_cloud_range
    #====================================================================================================#
    
    onnx_simp = generate_default_onnx(detector, BATCH_SIZE, MAX_N_VOXELS, 
                                      MAX_POINTS_PER_VOXEL, N_CHANNEL_POINT)
    
    modify_onnx(onnx_simp, BATCH_SIZE, MAX_N_POINTS, 
                N_CHANNEL_POINT, MAX_N_VOXELS, MAX_POINTS_PER_VOXEL,
                POINT_CLOUD_RANGE, N_CHANNEL_PILLAR, VOXEL_SIZE)

    print('onnx success')
    
if __name__ == "__main__":
    main()
