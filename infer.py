import paddle.fluid as fluid
from reader import *

def init_model(model_path='infer_model1'):
    # 创建执行器
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # 从模型中获取预测程序、输入数据名称列表、分类器
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=model_path, executor=exe)
    return exe, infer_program, feeded_var_names, target_var

def recognize_sound(wav_file_path, exe, infer_program, feeded_var_names, target_var):
    wav_file = load_wav(wav_file_path)
    result = exe.run(program=infer_program,
                     feed={feeded_var_names[0]: wav_file},
                     fetch_list=target_var)
    lab = np.argsort(result)
    probability = result[0][0][lab]
    label = lab[0][0][-1]
    label_name = label_name_dict[label]
    return probability, label_name

