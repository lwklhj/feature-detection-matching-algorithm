from superglue import SuperGlue
import torch
import os
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER

def main():
    dir = os.path.dirname(__file__)

    # Load model
    model_weights_path = os.path.join(dir, "weights/superglue_indoor.pth")
    model = SuperGlue(SuperGlue.default_config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.load_state_dict(torch.load(model_weights_path, map_location = device))

    # set the model to inference mode
    model.eval()

    batch_size = 1
    num_keypoints = 1024

    kpts0 = torch.randn(batch_size, num_keypoints, 2)
    kpts1 = torch.randn(batch_size, num_keypoints, 2)

    scores0 = torch.randn(batch_size, num_keypoints)
    scores1 = torch.randn(batch_size, num_keypoints)

    desc0 = torch.randn(batch_size, 256, num_keypoints)
    desc1 = torch.randn(batch_size, 256, num_keypoints)

    # onnx_filename = os.path.join(dir, weights.split("/")[-1].split(".")[0] + ".onnx")
    onnx_filename = model_weights_path.replace(".pth", ".onnx")

    # Export the model
    # export_options = torch.onnx.ExportOptions(dynamic_shapes=False)
    # export_output = torch.onnx.dynamo_export(model, kpts0, scores0, desc0, kpts1, scores1, desc1, export_options = export_options)
    # export_output.save(onnx_filename)

    torch.onnx.export(model,  # model being run
                    (kpts0, scores0, desc0, kpts1, scores1, desc1),  # model input (or a tuple for multiple inputs)
                    onnx_filename,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=17,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=["keypoints0",  # batch x feature_number0 x 2
                                    "scores0",  # batch x feature_number0
                                    "descriptors0",  # batch x feature_dims x feature_number0
                                    "keypoints1",  # batch x feature_number1 x 2
                                    "scores1",  # batch x feature_number1
                                    "descriptors1",  # batch x feature_dims x feature_number1
                                ],  # the model input names
                    output_names=["scores"],  # the model output names
                    dynamic_axes={'keypoints0': {1: 'feature_number_0'},
                                    'scores0': {1: 'feature_number_0'},
                                    'descriptors0': {2: 'feature_number_0'},
                                    'keypoints1': {1: 'feature_number_1'},
                                    'scores1': {1: 'feature_number_1'},
                                    'descriptors1': {2: 'feature_number_1'},
                                    },  # dynamic model input names
                    )

    # # check onnx model
    # onnx_model = onnx.load(onnx_filename)
    # # convert model
    # model_simp, check = onnxsim.simplify(onnx_model)
    # onnx.save(model_simp, onnx_filename)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.checker.check_model(model_simp)
    
    # Tensorrt engine conversion
    profile = Profile()
    profile.add("keypoints0", min=(1, 1, 2), opt=(1,512,2), max=(1,2048,2))
    profile.add("keypoints1", min=(1, 1, 2), opt=(1,512,2), max=(1,2048,2))
    profile.add("scores0", min=(1, 1), opt=(1,512), max=(1,2048))
    profile.add("scores1", min=(1, 1), opt=(1,512), max=(1,2048))
    profile.add("descriptors0", min=(1, 256, 1), opt=(1,256,512), max=(1,256,2048))
    profile.add("descriptors1", min=(1, 256, 1), opt=(1,256,512), max=(1,256,2048))
    engine = engine_from_network(
        network_from_onnx_path(onnx_filename), config=CreateConfig(profiles=[profile])
    )
    save_engine(engine, model_weights_path.replace(".pth", ".engine"))

if __name__ == "__main__":
    main()