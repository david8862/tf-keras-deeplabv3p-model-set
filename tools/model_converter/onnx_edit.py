#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to modify converted Deeplabv3p ONNX model to remove "Transpose" OP
at head. This can be used when you apply "--inputs_as_nchw" in keras to onnx
convert, so that you can get a full "NCHW" layout ONNX model for better PyTorch
style compatibility
"""
import os, sys, argparse
import onnx


def onnx_edit(input_model, output_model):
    onnx_model = onnx.load(input_model)
    graph = onnx_model.graph
    node  = graph.node

    # check if node[-2] is "Transpose", which we hope to delete
    assert node[-2].op_type == 'Transpose', 'model structure error, not found Transpose OP in tail.'

    # delete "Transpose", now node[-2] should be "Resize"
    graph.node.remove(node[-2])

    # check & connect last 2 OPs (Softmax & Resize) after delete "Transpose"
    assert node[-2].op_type == 'Resize' and node[-1].op_type == 'Softmax', 'model structure error, not found needed OP in tail.'
    assert len(node[-1].input) == 1 and len(node[-2].output) == 1, 'invalid OP input/output number.'
    node[-1].input[0] = node[-2].output[0]

    # check & update graph output shape from NHWC to NCHW, since we delete a "Transpose"
    assert len(graph.output) == 1, 'invalid model output number.'
    output_shape = graph.output[0].type.tensor_type.shape.dim

    # switch from NHWC to NCHW
    output_height = output_shape[1].dim_value
    output_width = output_shape[2].dim_value
    output_channel = output_shape[3].dim_value

    output_shape[1].dim_value = output_channel
    output_shape[2].dim_value = output_height
    output_shape[3].dim_value = output_width

    # change "Softmax" axis to 1 (channel dim in new NCHW layout)
    assert len(node[-1].attribute) == 1 and node[-1].attribute[0].name == 'axis', 'invalid attribute for Softmax.'
    node[-1].attribute[0].i = 1  # i is axis value


    # save changed model
    #graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    #info_model = onnx.helper.make_model(graph)
    #onnx_model = onnx.shape_inference.infer_shapes(info_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_model)
    print('Done.')



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='edit Deeplabv3p onnx model to delete the Transpose OP at head')
    parser.add_argument('--input_model', help='input Deeplabv3p onnx model file to edit', type=str, required=True)
    parser.add_argument('--output_model', help='output onnx model file to save', type=str, required=True)

    args = parser.parse_args()

    onnx_edit(args.input_model, args.output_model)


if __name__ == "__main__":
    main()
