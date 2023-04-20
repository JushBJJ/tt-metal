#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <iterator>
#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <queue>

#include "dtx/dtx.hpp"
#include "dtx/util_vector_of_ints.hpp"
#include "dtx/util.hpp"
#include "dtx/dtx_passes.hpp"

using namespace std;

bool test_GenerateAddresses() {
    bool pass = true;
    bool DEBUG = true;

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("consumer", 1);

    // Producer node: 1 tensor.
    node0->groups[0]->shape = {40};
    node0->groups[0]->address = 0;
    node0->groups[0]->core = {2,3};

    // Transformation #1: vertical blocks
    node1->groups[0]->shape = {40};
    node1->groups[0]->address = 0;
    node1->groups[0]->core = {5,6};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 0,  new DTXTensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 0,  new DTXTensor({0}, {20}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    if (DEBUG) dtx->print();

    pass = generate_transfer_addresses(dtx);

    return pass;
}

bool run_DTX_reverse_transformations_test_0(int DEBUG) {

    TransformationNode * node0 = new TransformationNode("producer", 2);
    TransformationNode * node1 = new TransformationNode("tx1", 2);

    // NODE 0:
    node0->groups[0]->shape = {40};
    node0->groups[1]->shape = {40};

    // NODE 1:
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 1, new DTXTensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 1, new DTXTensor({0}, {20}))   );

    node1->groups[1]->shape = {40};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 0, new DTXTensor({20}, {40}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 0,  new DTXTensor({0}, {20}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    DataTransformations * backwards = reverse_transformations(dtx);

    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 1, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 1, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 1, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 1, new DTXTensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 0, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 0, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 0, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 0, new DTXTensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;
    */

   return true;
}

bool run_DTX_reverse_transformations_test_1(int DEBUG) {

    TransformationNode * node0 = new TransformationNode("producer", 2);
    TransformationNode * node1 = new TransformationNode("tx1", 3);


    // NODE 0:
    node0->groups[0]->shape = {100};
    node0->groups[1]->shape = {200};

    // NODE 1:
    node1->groups[0]->shape = {20};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({60}, {70}),  0, new DTXTensor({0},  {10}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({70}, {80}),  0, new DTXTensor({10}, {20}))   );

    // NODE 1:
    node1->groups[1]->shape = {30};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({130}, {140}), 1, new DTXTensor({0},  {10}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({140}, {150}), 1, new DTXTensor({10}, {20}))  );

    node1->groups[2]->shape = {40};
    node1->groups[2]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({80},  {90}),  0, new DTXTensor({0},  {10}))  );
    node1->groups[2]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({160}, {170}), 1, new DTXTensor({10}, {20}))  );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    DataTransformations * backwards = reverse_transformations(dtx);


    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 1, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 1, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 1, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 1, new DTXTensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 0, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 0, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 0, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 0, new DTXTensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;
    */

   return true;
}

bool run_DTX_reverse_transformations_test_2(int DEBUG) {
    bool pass = true;

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);
    TransformationNode * node2 = new TransformationNode("tx2", 1);

    // NODE 0:
    node0->groups[0]->shape = {40};

    // NODE 1:
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({10}, {20}),  0, new DTXTensor({0},  {10}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {10}),  0, new DTXTensor({10}, {20}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({30}, {40}),  0, new DTXTensor({20}, {30}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {30}),  0, new DTXTensor({30}, {40}))   );

    // NODE 2:
    node2->groups[0]->shape = {40};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {30}),  0, new DTXTensor({0},  {10}))  );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({30}, {40}),  0, new DTXTensor({10}, {20}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {10}),  0, new DTXTensor({20}, {30}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({10}, {20}),  0, new DTXTensor({30}, {40}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);
    dtx->transformations.push_back(node2);

    dtx->print();
    pass &= collapse_transformations(dtx);
    dtx->print();

    DataTransformations * backwards = reverse_transformations(dtx);

    //backwards->print();
    pass &= collapse_transformations(backwards);
    backwards->print();
    pass &= generate_transfer_addresses(backwards);

    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 1, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 1, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 1, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 1, new DTXTensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 0, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 0, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 0, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 0, new DTXTensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;
    */

   return true;
}

bool run_DTX_reverse_transformations_test_3(int DEBUG) {
    bool pass = true;

    DataTransformations * dtx = new DataTransformations();

    // Create producer node
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {2,3,4};
    dtx->transformations.push_back(node0);

    pass &= transpose_yz(dtx);
    pass &= transpose_xy(dtx);
    dtx->print();

    DataTransformations * backwards = reverse_transformations(dtx);
    backwards->print();
    return pass;
}

bool test_DTX_reverse_transformations() {
    bool DEBUG = true;
    bool pass = true;

    pass &= run_DTX_reverse_transformations_test_0(DEBUG);
    pass &= run_DTX_reverse_transformations_test_1(DEBUG);
    pass &= run_DTX_reverse_transformations_test_2(DEBUG);
    pass &= run_DTX_reverse_transformations_test_3(DEBUG);
    return pass;
}

bool test_generate_sliced_ranges_helper_functions() {
    bool pass = true;

    // Part 1 - Test Generated sliced ranges
    vector<vector<vector<int>>> ranges = generate_sliced_ranges({10,10}, {2,2});

    // Part 2 - Test Generate list of cores based on range
    vector<vector<int>> list_of_cores;
    list_of_cores = generate_list_of_cores_based_on_range({0,0}, {0,0});
    list_of_cores = generate_list_of_cores_based_on_range({0,0}, {0,1});
    list_of_cores = generate_list_of_cores_based_on_range({0,0}, {3,3});
    list_of_cores = generate_list_of_cores_based_on_range({2,2}, {5,5});
    list_of_cores = generate_list_of_cores_based_on_range({3,2}, {3,5});
    list_of_cores = generate_list_of_cores_based_on_range({5,3}, {9,3});

    return pass;
}

bool test_pass_parallelize_generic_tensor_slice() {
    bool pass = true;

    // Part 3 - Test generic parallelization pass
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {20,20};
    dtx->transformations.push_back(node0);

    vector<int> slice_factors = {2,2};
    vector<int> cores_start   = {0,0};
    vector<int> cores_end     = {1,1};
    pass &= parallelize_generic_tensor_slice(dtx, slice_factors, cores_start, cores_end);

    return pass;
}


bool test_dim_order_counting_helper_function() {
    /*
    dim_order_counting({2,3},   {0,1});   // row major
    dim_order_counting({2,3},   {1,0});   // col major
    dim_order_counting({2,3,4}, {0,1,2}); // row major
    dim_order_counting({2,3,4}, {0,2,1}); // col major
    //dim_order_counting({2,3,4}, {1,2,0}); // sticks major, then row major after
    //dim_order_counting({2,3,4}, {2,1,0}); // sticks major, then col major after
    dim_order_counting({2,3,4,5}, {0,1,2,3}); // sticks major, then col major after
     */
    return true;
}

bool run_pass_tilize_and_store_test(vector<int> shape, vector<int> dim_order) {
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx->transformations.push_back(node0);
    bool pass = tilize_and_store(dtx, dim_order);
    return pass;
}

bool run_pass_tilize_and_store_test_0() {
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 2);
    node0->groups[0]->shape = {64,64};
    node0->groups[1]->shape = {128,128  };
    node0->groups[0]->core = {1,2};
    node0->groups[1]->core = {2,3};

    dtx->transformations.push_back(node0);
    bool pass = tilize_and_store(dtx, {0,1});
    dtx->print();
    return pass;
}


bool test_pass_tilize_and_store() {
    bool pass = true;

    // PART 1: The main helper function
    pass &= test_dim_order_counting_helper_function();

    // Test simple vector helper method
    /*
    cout << "dut: " << v2s(vector_pad_on_left({32,32}, 2, 0)) << endl;
    cout << "dut: " << v2s(vector_pad_on_left({32,32}, 2, 1)) << endl;
    cout << "dut: " << v2s(vector_pad_on_left({32,32}, 1, 0)) << endl;
    */

    // Test pass (producer groups == 1)
    //pass &= run_pass_tilize_and_store_test({64,64}, {1,0});
    //pass &= run_pass_tilize_and_store_test({128,128}, {1,0});
    //pass &= run_pass_tilize_and_store_test({2, 64, 64}, {2,1,0});

    // Test pass (producer groups > 1)
    pass &= run_pass_tilize_and_store_test_0();


    return pass;
}


bool test_transpose_xy() {
    TensorData * t = new TensorData({32,32});
    t->print();
    t->generate_csv("tensor1");

    return true;
}

bool test_tensor_evaluate() {
    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);

    // NODE 0:
    node0->groups[0]->shape = {40};

    // NODE 1:
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 1, new DTXTensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 1, new DTXTensor({0}, {20}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    TensorData * t = new TensorData({40});
    t->print();
    t->generate_csv("tensor1");

    //TensorData * t_out = dtx->evaluate(t_in);

    return true;
}

bool test_pass_transpose_xy() {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {40, 50};
    dtx->transformations.push_back(node0);
    pass &= transpose_xy(dtx);
    dtx->print();
    return pass;
}

bool run_transpose_yz(vector<int> shape) {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx->transformations.push_back(node0);
    pass &= transpose_yz(dtx);
    dtx->print();
    return pass;
}

bool test_pass_transpose_yz() {
    bool pass = true;

    //pass &= run_transpose_yz({2,3,4});
    pass &= run_transpose_yz({90, 2, 2, 100});
    pass &= run_transpose_yz({20, 21, 22, 2, 2, 100});
    return pass;

}

bool test_pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1() {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {10, 4, 4};
    dtx->transformations.push_back(node0);
    pass &= convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(dtx);
    dtx->print();
    return pass;
}

bool test_convert_abstract_tensor_to_channels_last_layout() {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {10, 4, 4};
    dtx->transformations.push_back(node0);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx);
    dtx->print();
    return pass;
}

bool test_channels_last_to_2D_matrix() {
    bool pass = true;

    /*
    // collapse transformation debug
    vector<int> shape = {5, 4,4};

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(dtx_right);
    pass &= row_major_memory_store(dtx_right);
    pass &= collapse_transformations(dtx_right);
    */



    // FULL TEST

    vector<int> shape = {5, 4,4};

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(dtx_right);
    pass &= row_major_memory_store(dtx_right);

    cout << "\n\nDTX_RIGHT" << endl;
    dtx_right->print();


    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx_left);

    cout << "\n\nDTX_LEFT" << endl;
    dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    cout << "\n\nDTX_COMBINED" << endl;
    combined->print();

    pass &= optimize_away_transpose(combined);
    cout << "\n\nDTX_OPTIMIZED" << endl;
    combined->print();

    pass &= collapse_transformations(combined);
    cout << "\n\nDTX_COLLAPSED" << endl;
    combined->print();
    pass &= generate_transfer_addresses(combined);
    combined->print();



    return pass;
}

bool test_channels_last_to_2D_matrix_conv1x1() {
    bool pass = true;


    // FULL TEST

    vector<int> shape = {5, 4,4};

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= convert_tensor_layout_CL1_to_2Dmatrix_conv1x1_s1(dtx_right);
    pass &= row_major_memory_store(dtx_right);

    cout << "\n\nDTX_RIGHT" << endl;
    dtx_right->print();


    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx_left);

    cout << "\n\nDTX_LEFT" << endl;
    dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    cout << "\n\nDTX_COMBINED" << endl;
    combined->print();

    pass &= optimize_away_transpose(combined);
    cout << "\n\nDTX_OPTIMIZED" << endl;
    combined->print();

    pass &= collapse_transformations(combined);
    cout << "\n\nDTX_COLLAPSED" << endl;
    combined->print();
    pass &= generate_transfer_addresses(combined);
    combined->print();



    return pass;
}

bool test_high_level_pass_and_evaluate() {
    vector<int> shape = {2, 2, 2};
    auto dtx = simple_high_level_pass(shape);
    vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
    vector<float> data_transformed = evaluate(data, dtx);
    vector<float> golden_data = {1, 2, 5, 6, 3, 4, 7, 8};
    return data_transformed == golden_data;
}

bool test_padding_pass_(vector<int> shape, vector<int> pad_to_nearest, vector<float> input_data, vector<float> golden_data) {
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    bool pass = row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= pad_2d_matrix(dtx_right, pad_to_nearest);
    //dtx_right->print();
    //exit(1);
    pass &= row_major_memory_store(dtx_right);
    //dtx_right->print();
    //exit(1);
    //pass &= collapse_transformations(dtx_right);
    //dtx_right->print();
    //exit(1);
    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    vector<float> data_transformed = evaluate(input_data, combined);
    return data_transformed == golden_data;
}

bool test_padding_pass() {
    bool pass = true;
    vector<int> shape = {1, 2, 2};
    vector<float> input_data_2_2 = {1, 2, 3, 4};
    // list of tests - pad to nearest, golden data
    vector<tuple<vector<int>, vector<float>>> tests_2_2 = {
        { {4,4}, {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
        { {3,4}, {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0} },
        { {4,3}, {1, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0} },
    };
    for (auto & t : tests_2_2) {
        auto pad_to_nearest = std::get<0>(t);
        auto golden_data = std::get<1>(t);
        pass &= test_padding_pass_(shape, pad_to_nearest, input_data_2_2, golden_data);
        if(pass) {
            std::cout << "Passed test with shape = ";
        }
        else {
            std::cout << "Failed test with shape = ";
        }
        std::cout << v2s(shape) << " , pad to nearest = " << v2s(pad_to_nearest) << std::endl;
        if(!pass) exit(1);
    }
    return pass;
}

bool test_block_2d_matrix_pass_(vector<int> shape, vector<int> block_shape, vector<int> dim_order,
                            vector<float> input_data, vector<float> golden_data) {
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    bool pass = row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= block_2d_matrix(dtx_right, dim_order, block_shape);
    pass &= row_major_memory_store(dtx_right);
    //dtx_right->print();
    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    vector<float> data_transformed = evaluate(input_data, combined);
    return data_transformed == golden_data;
}

bool test_block_2d_matrix_pass() {
    bool pass = true;
    vector<int> shape = {1, 4, 4};
    vector<float> input_data_4_4 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // list of tests - block shape, dim order, golden data
    vector<tuple<vector<int>, vector<int>, vector<float>>> tests_4_4 = {
        { {4,4}, {0,1,2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} },
        { {2,4}, {0,1,2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} },
        { {1,4}, {0,1,2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} },
        { {4,1}, {0,1,2}, {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16} },
        { {4,2}, {0,1,2}, {1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16} },
        { {2,2}, {0,1,2}, {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16} },
        { {2,2}, {0,2,1}, {1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16} },

    };
    for (auto & t : tests_4_4) {
        auto block_shape = std::get<0>(t);
        auto dim_order = std::get<1>(t);
        auto golden_data = std::get<2>(t);
        pass &= test_block_2d_matrix_pass_(shape, block_shape, dim_order, input_data_4_4, golden_data);
        if(pass) {
            std::cout << "Passed test with shape = ";
        }
        else {
            std::cout << "Failed test with shape = ";
        }
        std::cout << v2s(shape) << " , block shape = " << v2s(block_shape) << " , dim order = " << v2s(dim_order) << std::endl;
    }
    // Testing shape with x != y
    shape = {1, 2, 6};
    vector<float> input_data_2_6 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // list of tests - block shape, dim order, golden data
    vector<tuple<vector<int>, vector<int>, vector<float>>> tests_2_6 = {
        { {2,6}, {0,1,2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} },
        { {1,6}, {0,1,2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} },
        { {2,2}, {0,1,2}, {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12} },
        { {2,3}, {0,1,2}, {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12} },
        { {1,3}, {0,1,2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} },
        { {1,3}, {0,2,1}, {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12} },

    };
    for (auto & t : tests_2_6) {
        auto block_shape = std::get<0>(t);
        auto dim_order = std::get<1>(t);
        auto golden_data = std::get<2>(t);
        pass &= test_block_2d_matrix_pass_(shape, block_shape, dim_order, input_data_2_6, golden_data);
        if(pass) {
            std::cout << "Passed test with shape = ";
        }
        else {
            std::cout << "Failed test with shape = ";
        }
        std::cout << v2s(shape) << " , block shape = " << v2s(block_shape) << " , dim order = " << v2s(dim_order) << std::endl;
    }
    return pass;

}

bool test_pad_and_block_passes_(vector<int> shape, vector<int> pad_to_nearest, vector<int> block_shape, vector<int> dim_order,
                            vector<float> input_data, vector<float> golden_data) {
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    bool pass = row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= pad_2d_matrix(dtx_right, pad_to_nearest);
    pass &= block_2d_matrix(dtx_right, dim_order, block_shape);
    //dtx_right->print();
    pass &= row_major_memory_store(dtx_right);
    //dtx_right->print();
    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    vector<float> data_transformed = evaluate(input_data, combined);
    return data_transformed == golden_data;
}
bool test_pad_and_block_passes() {
    bool pass = true;
    vector<int> shape = {1, 2, 2};
    vector<float> input_data_2_2 = {1, 2, 3, 4};
    // list of tests - pad to nearest, block shape, dim order, golden data
    vector<tuple<vector<int>, vector<int>, vector<int>, vector<float>>> tests_2_2 = {
        { {4,4}, {4,4}, {0,1,2}, {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
        { {4,4}, {2,4}, {0,1,2}, {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
        { {4,4}, {2,2}, {0,1,2}, {1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} },
    };
    for (auto & t : tests_2_2) {
        auto pad_to_nearest = std::get<0>(t);
        auto block_shape = std::get<1>(t);
        auto dim_order = std::get<2>(t);
        auto golden_data = std::get<3>(t);
        pass &= test_pad_and_block_passes_(shape, pad_to_nearest, block_shape, dim_order, input_data_2_2, golden_data);
        if(pass) {
            std::cout << "Passed test with shape = ";
        }
        else {
            std::cout << "Failed test with shape = ";
        }
        std::cout << v2s(shape) <<
            " , pad to nearest = " << v2s(pad_to_nearest) <<
            " , block shape = " << v2s(block_shape) <<
            " , dim order = " << v2s(dim_order) <<
            std::endl;
        if (!pass) exit(1);
    }
    return pass;
}

bool test_conv_transform_pass_and_evaluate() {
    vector<int> shape = {2, 2, 2};
    bool pass = true;
    auto dtx = conv_transform(shape, {1,1,1,1,0,0}, false);
    vector<float> data_1_1 = {1, 2, 3, 4, 5, 6, 7, 8};
    vector<float> data_transformed_1_1 = evaluate(data_1_1, dtx);
    vector<float> golden_data_1_1 = {1, 2, 3, 4, 5, 6, 7, 8};
    pass &= data_transformed_1_1 == golden_data_1_1;
    if(!pass) {
        std::cout << "1x1 conv transform failed." << std::endl;
    }
    vector<int> shape2 = {2, 3, 3};
    auto dtx2 = conv_transform(shape2, {3,3,1,1,0,0}, false);
    //vector<float> data_3_3 = {1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24};
    vector<float> data_3_3 = {1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18};
    vector<float> data_transformed_3_3 = evaluate(data_3_3, dtx2);
    for(uint i = 0; i < data_transformed_3_3.size(); i++) {
        std::cout << "data transformed at i, " << i << ", = " << data_transformed_3_3[i] << std::endl;
    }
    // vector<float> golden_data_3_3 = {1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21,
    //                                    4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24};
    vector<float> golden_data_3_3 ={1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18};
    pass &= data_transformed_3_3 == golden_data_3_3;
    if (!pass) {
        std::cout << "3x3 conv transform failed." << std::endl;
    }
    return pass;
}

void run_dtx_tests() {
    bool pass = true;

    cout << "==================================================================" << endl;
    cout << "                         Starting DTX TESTs                       " << endl;
    cout << "==================================================================" << endl;

    // pass &= test_GenerateAddresses();
    // printf("test_GenerateAddresses - %d\n\n", pass);

    //pass &= test_DTX_reverse_transformations();
    //printf("test_DTX_reverse_transformations - %d\n\n", pass);

    // pass &= test_generate_sliced_ranges_helper_functions();
    // printf("test_generate_sliced_ranges_helper_functions - %d\n\n", pass);

    // pass &= test_pass_parallelize_generic_tensor_slice();
    // printf("test_pass_parallelize_generic_tensor_slice - %d\n\n", pass);

    // pass &= test_pass_tilize_and_store();
    // printf("test_pass_tilize_and_store - %d\n\n", pass);

    //pass &= test_transpose_xy();
    //printf("test_transpose_xy - %d\n\n", pass);

    //pass &= test_tensor_evaluate();
    //printf("test_tensor_evaluate - %d\n\n", pass);

    // pass &= test_pass_transpose_xy();
    // printf("test_pass_transpose_xy - %d\n\n", pass);

    // pass &= test_pass_transpose_yz();
    // printf("test_pass_transpose_yz - %d\n\n", pass);

    //pass &= test_pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1();
    //printf("test_pass_transpose_xy - %d\n\n", pass);

    //pass &= test_convert_abstract_tensor_to_channels_last_layout();         // TO DO: generalize rank
    //printf("test_pass_convert_abstract_tensor_to_channels_last_layout - %d\n\n", pass);

    // In progress
    pass &= test_channels_last_to_2D_matrix();
    printf("test_channels_last_to_2D_matrix - %d\n\n", pass);

    pass &= test_channels_last_to_2D_matrix_conv1x1();
    printf("test_channels_last_to_2D_matrix_conv1x1 - %d\n\n", pass);

    pass &= test_high_level_pass_and_evaluate();
    printf("test_high_level_pass_and_evaluate - %d\n\n", pass);

    pass &= test_block_2d_matrix_pass();
    printf("test_block_2d_matrix_pass - %d\n\n", pass);

    pass &= test_padding_pass();
    printf("test_pad_2d_matrix_pass - %d\n\n", pass);

    pass &= test_pad_and_block_passes();
    printf("test_pad_and_block_passes - %d\n\n", pass);

    pass &= test_conv_transform_pass_and_evaluate();
    printf("test_conv_transform_pass - %d\n\n", pass);

    if (pass == true) cout << "\nTESTS PASSED\n\n\n" << endl;
    else cout << "TESTS FAILED\n\n\n" << endl;
}

// ===============================================================
// ===============================================================

int main(int argc, char** argv) {

    // Run all Data Transformation Tests
    run_dtx_tests();
}
