#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"
#include "dtx_passes.hpp"


bool convert_tensor_layout_CL1_to_2Dmatrix(DataTransformations * dtx,
                                        vector<int> conv_params) {
    /*
    Notes:
    Should not support groups (I think). This transformation has to be applied before parallelizations.
    What if it's applied to a DRAM swizzled layout - need to think about that.

    Note: Input activatins need to already be padded. Padding not supported yet.
    // int pad = 1; // all around
    */

    bool DEBUG = true;
    bool pass = true;
    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix - START" << endl;

    // First add the 2 required transposes
    pass &= transpose_yz(dtx);
    pass &= transpose_xy(dtx);

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1", producer->groups.size());
    dtx->transformations.push_back(consumer);

    // Calculate producer shape
    vector<int> producer_shape = producer->groups[0]->shape;
    int rank = producer_shape.size();
    assert(rank == 3); // TODO: generalize for num batches > 1
    // Calculate the consumer shape
    int R = conv_params[0];
    int S = conv_params[1];
    int U = conv_params[2];
    int V = conv_params[3];
    int Pad_H = conv_params[4];
    int Pad_W = conv_params[5];
    assert(R == S);
    assert(U == V);
    int consumer_shape_x = producer_shape[X(rank)] * R * S;
    int conv_output_h = ((producer_shape[Z(rank)] - R + (2 * Pad_H)) / U)+ 1;
    int conv_output_w = ((producer_shape[Y(rank)] - S + (2 * Pad_W)) / V) + 1;
    std::cout << "consumer_output_h=" << conv_output_h << std::endl;
    std::cout << "consumer_output_w=" << conv_output_w << std::endl;
    int consumer_shape_y = conv_output_h * conv_output_w;
    assert(consumer_shape_y > 0);
    consumer->groups[0]->shape = {1, consumer_shape_y, consumer_shape_x};
    vector<int> consumer_shape = consumer->groups[0]->shape;

    int conv_input_x = producer_shape[Y(rank)];
    int conv_input_y = producer_shape[Z(rank)];
    int conv_input_z = producer_shape[X(rank)];
    int consumer_y = 0;
    int consumer_x = 0;
    for(int x = 0; x <= conv_input_x-S; x+=V) {
        for(int y = 0; y <= conv_input_y-R; y+=U) {
            for(int s = 0; s < S; s++) {
                for(int r = 0; r < R; r++) {
                    // Producer str/end
                    vector<int> producer_str = { y + r, x + s, 0};
                    vector<int> producer_end = { y + r, x + s, conv_input_z-1};

                    vector<int> consumer_str = {0, consumer_y, consumer_x};
                    vector<int> consumer_end = {0, consumer_y, consumer_x + conv_input_z-1};

                    TensorPair * tp = new TensorPair(new DTXTensor({producer_str}, {producer_end}),
                                            0,
                                            new DTXTensor({consumer_str}, {consumer_end}));
                    consumer->groups[0]->tensor_pairs.push_back(tp);

                    if (DEBUG) cout << sp(6) << "src = " << v2s(producer_str) << "-" << v2s(producer_end) << " ==> " << v2s(consumer_str) << "-" << v2s(consumer_end) << endl;

                    consumer_x += conv_input_z; // length of channel
                }
            }
            consumer_y++;
            consumer_x = 0;
        }
    }
    if (DEBUG) cout << "\n\nPASS: convert_tensor_layout_CL1_to_2Dmatrix - END\n\n" << endl;
    return true;
}
