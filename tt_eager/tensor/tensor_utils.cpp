// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor_utils.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tensor/borrowed_buffer.hpp"
#include "tensor/borrowed_buffer_functions.hpp"

namespace tt {

namespace tt_metal {

    std::vector<int> compute_strides(Shape shape) {
            auto num_elements = compute_volume(shape);
            std::vector<int> strides;
            for (std::int32_t index = 0; index < shape.rank(); index++) {
                num_elements /= shape[index];
                strides.push_back(num_elements);
            }
            return strides;
        }

    int compute_flat_input_index(vector<int> indices, vector<int> strides) {
            int flat_index = 0;
            for (auto i = 0; i < indices.size(); i++) {
                flat_index += indices[i] * strides[i];
            }
            return flat_index;
        };

    template <typename T>
    Tensor to_weight_special_padding_tile_layout(const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
        auto w_shape = conv_weight_tensor.get_legacy_shape();
        auto input_buffer = owned_buffer::get_as<T>(conv_weight_tensor);
        uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
        uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
        auto weight_matrix_cols = w_shape[0];
        // width padding
        if(weight_matrix_cols%in1_block_w_datums != 0) {
            weight_matrix_cols = (uint32_t) std::ceil( (double) weight_matrix_cols / (double) in1_block_w_datums ) * in1_block_w_datums;
        }
        // height padding
        assert(in1_block_h_datums >= w_shape[1]*w_shape[3]);
        uint32_t block_height_padding = in1_block_h_datums - (w_shape[1]*w_shape[3]);
        auto weight_matrix_rows = ((w_shape[1]*w_shape[3]) + block_height_padding)*w_shape[2];
        Shape output_shape = {1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(compute_volume(output_shape));
        for(auto r = 0; r < w_shape[2]; r++) {
            for(auto s = 0; s < w_shape[3]; s++) {
                for(auto c = 0; c < w_shape[1]; c++) {
                    for(auto k = 0; k < w_shape[0]; k++) {
                        auto matrix_idx = k + c * weight_matrix_cols + s * w_shape[1] * weight_matrix_cols + r * ((w_shape[3] * w_shape[1]) + block_height_padding) * weight_matrix_cols;
			auto idx = k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
			output_buffer[matrix_idx] = input_buffer[idx];
                    }
                }
            }
        }
        if constexpr (std::is_same<T, float>::value) {
            if (output_dtype == DataType::BFLOAT8_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
            if (output_dtype == DataType::BFLOAT4_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
        } else {
            TT_ASSERT((output_dtype != DataType::BFLOAT8_B) || (output_dtype != DataType::BFLOAT4_B));
        }
        auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
        return rm_tensor.to(Layout::TILE);
    }


    template <typename T>
    Tensor to_weight_tile_layout(const Tensor& conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype) {
        auto w_shape = conv_weight_tensor.get_legacy_shape();
        auto input_buffer = owned_buffer::get_as<T>(conv_weight_tensor);
        auto weight_matrix_cols = w_shape[0];
        // width padding
        uint32_t in1_block_w_datums = in1_block_w * constants::TILE_WIDTH;
        if(weight_matrix_cols%in1_block_w_datums != 0) {
            weight_matrix_cols = (uint32_t) std::ceil( (double) weight_matrix_cols / (double) in1_block_w_datums ) * in1_block_w_datums;
        }
        // height padding
        auto weight_matrix_rows = w_shape[1]*w_shape[2]*w_shape[3];
        uint32_t in1_block_h_datums = in1_block_h * constants::TILE_HEIGHT;
        if (weight_matrix_rows % in1_block_h_datums != 0) {
            weight_matrix_rows = (uint32_t) std::ceil( (double) weight_matrix_rows / (double) in1_block_h_datums ) * in1_block_h_datums;
        }
        Shape output_shape = {1, 1, weight_matrix_rows, weight_matrix_cols};
        auto output_buffer = owned_buffer::create<T>(compute_volume(output_shape));
        for(auto r = 0; r < w_shape[2]; r++) {
            for(auto s = 0; s < w_shape[3]; s++) {
                for(auto c = 0; c < w_shape[1]; c++) {
                    for(auto k = 0; k < w_shape[0]; k++) {
                        auto matrix_idx = k + c * weight_matrix_cols + s * w_shape[1] * weight_matrix_cols + r * w_shape[3] * w_shape[1] * weight_matrix_cols;
                        auto idx = k * w_shape[1] * w_shape[2] * w_shape[3] + c * w_shape[2] * w_shape[3] + r * w_shape[3] + s;
                        output_buffer[matrix_idx] = input_buffer[idx];
                    }
                }
            }
        }
        if constexpr (std::is_same<T, float>::value) {
            if (output_dtype == DataType::BFLOAT8_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
            if (output_dtype == DataType::BFLOAT4_B) {
                auto output_float_data = output_buffer.get();
                auto output_packed_data = pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false);
                auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
                auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
                return rm_tensor.to(Layout::TILE);
            }
        } else {
            TT_ASSERT((output_dtype != DataType::BFLOAT8_B) || (output_dtype != DataType::BFLOAT4_B));
        }
        auto rm_tensor = Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_shape, output_dtype, Layout::ROW_MAJOR);
        return rm_tensor.to(Layout::TILE);
    }

    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, std::optional<DataType> output_dtype) {
        TT_ASSERT(conv_weight_tensor.get_layout() == Layout::ROW_MAJOR && "Convolution weights should be in row major layout for conversion to tilized layout.");
        const static std::map<DataType, std::function<Tensor(const Tensor &, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype)>> to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_tile_layout<float>},
            {DataType::UINT32, &to_weight_tile_layout<uint32_t>},
        };
        if (output_dtype.has_value()) {
            if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
                TT_ASSERT(conv_weight_tensor.get_dtype() == DataType::FLOAT32);
            } else {
                TT_ASSERT(conv_weight_tensor.get_dtype() == conv_weight_tensor.get_dtype());
            }
        }
        return to_w_tile_layout_map.at(conv_weight_tensor.get_dtype())(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype.value_or(conv_weight_tensor.get_dtype()));
    }

    // Converts convolution weights to tilized 2d matrix layout.
    // Returns a new tensor with layout=Tile
    Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(Tensor conv_weight_tensor, uint32_t in1_block_h, uint32_t in1_block_w, std::optional<DataType> output_dtype) {
        TT_ASSERT(conv_weight_tensor.get_layout() == Layout::ROW_MAJOR && "Convolution weights should be in row major layout for conversion to tilized layout.");
        const static std::map<DataType, std::function<Tensor(const Tensor &, uint32_t in1_block_h, uint32_t in1_block_w, DataType output_dtype)>> to_w_tile_layout_map = {
            {DataType::BFLOAT16, &to_weight_special_padding_tile_layout<bfloat16>},
            {DataType::FLOAT32, &to_weight_special_padding_tile_layout<float>},
            {DataType::UINT32, &to_weight_special_padding_tile_layout<uint32_t>}
        };
        if (output_dtype.has_value()) {
            if (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) {
                TT_ASSERT(conv_weight_tensor.get_dtype() == DataType::FLOAT32);
            } else {
                TT_ASSERT(conv_weight_tensor.get_dtype() == conv_weight_tensor.get_dtype());
            }
        }
        return to_w_tile_layout_map.at(conv_weight_tensor.get_dtype())(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype.value_or(conv_weight_tensor.get_dtype()));
    }

    // Converts convolution weights to grouped layout with padded zeros
    // taps
    Tensor convert_conv_weight_tensor_to_grouped_layout(Tensor conv_weight_tensor, uint32_t num_groups, DataType output_dtype) {

        std::cout << "DEBUG: " << "num_groups=" << num_groups << std::endl;

        TT_ASSERT(conv_weight_tensor.get_layout() == Layout::ROW_MAJOR && "Convolution weights should be in row major layout for adding the required padding");

        // Define output tensor shape. This is going to be channel dimension of weight tensor * num_groups - this value should match number of input channels being convolved with the weight tensor
        auto original_conv_weight_tensor_shape_test = conv_weight_tensor.get_shape();
        Shape original_conv_weight_tensor_shape = {original_conv_weight_tensor_shape_test[0], original_conv_weight_tensor_shape_test[1], original_conv_weight_tensor_shape_test[2], original_conv_weight_tensor_shape_test[3]};
        Shape output_conv_weight_tensor_shape = {original_conv_weight_tensor_shape[0], original_conv_weight_tensor_shape[1] * num_groups, original_conv_weight_tensor_shape[2], original_conv_weight_tensor_shape[3]};

        std::cout << "DEBUG: original weight shape=" << original_conv_weight_tensor_shape[0] << "," << original_conv_weight_tensor_shape[1] << "," << original_conv_weight_tensor_shape[2] << "," << original_conv_weight_tensor_shape[3] << num_groups << std::endl;
        std::cout << "DEBUG: output weight shape=" << output_conv_weight_tensor_shape[0] << "," << output_conv_weight_tensor_shape[1] << "," << output_conv_weight_tensor_shape[2] << "," << output_conv_weight_tensor_shape[3] << num_groups << std::endl;


        // Create newly allocated buffer all initialized to 0
        int num_filters_per_group = original_conv_weight_tensor_shape[0] / num_groups;

        std::cout << "DEBUG: num filters per group=" << num_filters_per_group << std::endl;

        if (output_dtype == DataType::INT32) {
            std::cout << "DEBUG: int32" << std::endl;
            owned_buffer::Buffer<int32_t> output_buffer = owned_buffer::create<int32_t>(compute_volume(output_conv_weight_tensor_shape));
            std::cout << "DEBUG: create output buffer" << std::endl;
            const auto conv_weight_tensor_buffer = owned_buffer::get_as<int32_t>(conv_weight_tensor);
            std::cout << "DEBUG: got original weight buffer" << std::endl;
        } else if (output_dtype == DataType::FLOAT32) {
            std::cout << "DEBUG: float32" << std::endl;
            owned_buffer::Buffer<float> output_buffer = owned_buffer::create<float>(compute_volume(output_conv_weight_tensor_shape));
            std::cout << "DEBUG: create output buffer" << std::endl;
            const auto conv_weight_tensor_buffer = owned_buffer::get_as<float>(conv_weight_tensor);
            std::cout << "DEBUG: got original weight buffer" << std::endl;
        } else if (output_dtype == DataType::BFLOAT16) {
            // taps
            std::cout << "DEBUG: bfloat16" << std::endl;
            owned_buffer::Buffer<bfloat16> output_buffer = owned_buffer::create<bfloat16>(compute_volume(output_conv_weight_tensor_shape));
            std::cout << "DEBUG: create output buffer" << std::endl;
            auto conv_weight_tensor_buffer = borrowed_buffer::get_as<bfloat16>(conv_weight_tensor);
            std::cout << "DEBUG: got original weight buffer" << std::endl;

            int input_weight_n = original_conv_weight_tensor_shape[0];
            int input_weight_c = original_conv_weight_tensor_shape[1];
            int input_weight_h = original_conv_weight_tensor_shape[2];
            int input_weight_w = original_conv_weight_tensor_shape[3];

            std::cout << "DEBUG: input_weight_n=" << input_weight_n << std::endl;
            std::cout << "DEBUG: input_weight_c=" << input_weight_c << std::endl;
            std::cout << "DEBUG: input_weight_h=" << input_weight_h << std::endl;
            std::cout << "DEBUG: input_weight_w=" << input_weight_w << std::endl;

            for (int curr_batch_idx = 0; curr_batch_idx < input_weight_n; curr_batch_idx++) {
                int new_batch_idx = curr_batch_idx;
                int new_channel_start_idx = curr_batch_idx * input_weight_c;

                for (int j = 0; j < input_weight_c; j++) {
                    for (int k = 0; k < input_weight_h; k++) {
                        for (int m = 0; m < input_weight_w; m++) {
                            auto value_flat_input_index = compute_flat_input_index({curr_batch_idx, j, k, m}, compute_strides(original_conv_weight_tensor_shape));
                            auto value = conv_weight_tensor_buffer[value_flat_input_index];

                            auto new_channel_idx = new_channel_start_idx + j;
                            auto output_flat_input_index = compute_flat_input_index({new_batch_idx, new_channel_idx, k, m}, compute_strides(output_conv_weight_tensor_shape));
                            output_buffer[output_flat_input_index] = value;

                            std::cout << "DEBUG: i=" << curr_batch_idx << std::endl;
                            std::cout << "DEBUG: j=" << j << std::endl;
                            std::cout << "DEBUG: k=" << k << std::endl;
                            std::cout << "DEBUG: m=" << m << std::endl;
                            std::cout << "DEBUG: value_flat_input_index=" << value_flat_input_index << std::endl;
                            std::cout << "DEBUG: output_flat_input_index=" << output_flat_input_index << std::endl;
                        }
                    }
                }
            }

            std::cout << "DEBUG: creating return tensor" << std::endl;
            auto output_tensor = Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_conv_weight_tensor_shape, output_dtype, Layout::ROW_MAJOR);
            std::cout << "DEBUG: yay success!" << std::endl;
            return output_tensor;
        } else if (output_dtype == DataType::UINT16) {
            std::cout << "DEBUG: uint16" << std::endl;
            owned_buffer::Buffer<uint16_t> output_buffer = owned_buffer::create<uint16_t>(compute_volume(output_conv_weight_tensor_shape));
            std::cout << "DEBUG: create output buffer" << std::endl;
            const auto conv_weight_tensor_buffer = owned_buffer::get_as<uint16_t>(conv_weight_tensor);
            std::cout << "DEBUG: got original weight buffer" << std::endl;
        } else {
            std::cout << "DEBUG: uint32" << std::endl;
            owned_buffer::Buffer<uint32_t> output_buffer = owned_buffer::create<uint32_t>(compute_volume(output_conv_weight_tensor_shape));
            std::cout << "DEBUG: create output buffer" << std::endl;
            const auto conv_weight_tensor_buffer = owned_buffer::get_as<uint32_t>(conv_weight_tensor);
            std::cout << "DEBUG: got original weight buffer" << std::endl;
        }

        // Loop through the newly allocated tensor and slot in the original weight tensor values. The remaining values will be defaulted to 0
        //int group_idx = 0;
        //for (int i = 0; i < compute_volume(original_conv_weight_tensor_shape); i++) {
        //    std::cout << conv_weight_tensor.get_storage();
        //}

        // Move buffer ownership in tensor and return
        std::cout << "DEBUG: before creating a random output buffer" << std::endl;
        owned_buffer::Buffer<int32_t> output_buffer = owned_buffer::create<int32_t>(compute_volume(output_conv_weight_tensor_shape));
        std::cout << "DEBUG: before creating a random tensor" << std::endl;
        auto output_tensor = Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_conv_weight_tensor_shape, output_dtype, Layout::ROW_MAJOR);

        std::cout << "DEBUG: success!" << std::endl;
        return output_tensor;
    }

const Shape infer_dims_for_reshape(int N, int C, int H, int W, uint32_t old_volume) {
    vector<int> ns{N, C, H, W};
    int neg_idx = -1;
    for (int i = 0; i < ns.size(); i++) {
        if (ns[i] == -1) {
            TT_ASSERT(neg_idx == -1, "Only one -1 is allowed in reshape");
            neg_idx = i;
        } else {
            TT_ASSERT(ns[i] > 0, "New shape entries can only have -1 or positive values");
        }
    }

    switch (neg_idx) {
        case 0:
            TT_ASSERT(old_volume % C*H*W == 0);
            N = old_volume/(C*H*W);
            break;
        case 1:
            TT_ASSERT(old_volume % N*H*W == 0);
            C = old_volume/(N*H*W);
            break;
        case 2:
            TT_ASSERT(old_volume % N*C*W == 0);
            H = old_volume/(N*C*W);
            break;
        case 3:
            TT_ASSERT(old_volume % N*C*H == 0);
            W = old_volume/(N*C*H);
            break;
        case -1: // In case where there is no negative value in ns
            TT_ASSERT(N*C*H*W == old_volume);
            break;
        default:
            TT_ASSERT(false && "Unexpected neg_idx in reshape!");
    }

    return {(uint32_t)N, (uint32_t)C, (uint32_t)H, (uint32_t)W};
}

  bool is_arch_gs(const tt::ARCH& arch) {
    return arch == tt::ARCH::GRAYSKULL;
  }

  bool is_arch_whb0(const tt::ARCH& arch) {
    return arch == tt::ARCH::WORMHOLE_B0;
  }

  bool is_cpu_tensor(const Tensor& tensor) {
      return tensor.storage_type() == StorageType::OWNED || tensor.storage_type() == StorageType::BORROWED;
  }

  bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

Tensor get_device_tensor(Device* device, const Tensor& multi_device_tensor) {
    const auto& tensor_storage = std::get<MultiDeviceStorage>(multi_device_tensor.get_storage());
    if (tensor_storage.buffers.find(device->id()) != tensor_storage.buffers.end()) {
        return Tensor{
            DeviceStorage{tensor_storage.buffers.at(device->id())},
            multi_device_tensor.get_legacy_shape(),
            multi_device_tensor.get_dtype(),
            multi_device_tensor.get_layout()
        };
    }
    TT_THROW("Device not found in multi-device tensor");
}

bool is_multi_device_tensor(const Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE or tensor.storage_type() == StorageType::MULTI_DEVICE_HOST;
}


std::vector<Tensor> get_tensors_from_multi_device_storage(const Tensor& multi_device_tensor) {
    std::vector<ttnn::Tensor> tensors;
    if (multi_device_tensor.storage_type() == StorageType::MULTI_DEVICE) {
        const auto& tensor_storage = std::get<MultiDeviceStorage>(multi_device_tensor.get_storage());
        tensors = std::vector<ttnn::Tensor>(tensor_storage.buffers.size(), Tensor());
        for (auto& device_buf_pair : tensor_storage.buffers) {
            auto [device_id, buffer] = device_buf_pair;
            tensors[device_id] = Tensor{DeviceStorage{buffer}, tensor_storage.shapes.at(device_id), multi_device_tensor.get_dtype(), multi_device_tensor.get_layout()};
        }
        return tensors;
    } else if (multi_device_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST) {
        const auto& tensor_storage = std::get<MultiDeviceHostStorage>(multi_device_tensor.get_storage());
        for (int i = 0; i < tensor_storage.buffers.size(); ++i) {
            tensors.push_back(Tensor{
                OwnedStorage{tensor_storage.buffers[i]},
                tensor_storage.shapes[i],
                multi_device_tensor.get_dtype(),
                multi_device_tensor.get_layout()
            });
        }
    }
    else {
        TT_FATAL(false, "get_tensors_from_multi_device_storage only support multi device tensors");
    }
    return tensors;
}

DistributedTensorConfig get_distributed_tensor_config_from_tensor(const Tensor& tensor) {
    if (tensor.storage_type() == StorageType::MULTI_DEVICE) {
        const auto& tensor_storage = std::get<MultiDeviceStorage>(tensor.get_storage());
        return tensor_storage.strategy;
    }
    else if (tensor.storage_type() == StorageType::MULTI_DEVICE_HOST) {
        const auto& tensor_storage = std::get<MultiDeviceHostStorage>(tensor.get_storage());
        return tensor_storage.strategy;
    }
    TT_THROW("Tensor is not a multi-device tensor");
}

Tensor create_multi_device_tensor(const std::vector<Tensor>& tensors, StorageType storage_type, const DistributedTensorConfig& strategy) {
    if (tensors.empty()) {
        TT_THROW("Cannot create multi-device tensor with empty tensor list");
    }

    if (storage_type == StorageType::MULTI_DEVICE) {
        std::unordered_map<int, tt::tt_metal::Shape> shapes;
        std::unordered_map<int, DeviceBuffer> device_buffers;
        for (const auto& tensor : tensors) {
            Device* device = std::get<DeviceStorage>(tensor.get_storage()).buffer->device();
            device_buffers.insert({device->id(), std::get<DeviceStorage>(tensor.get_storage()).buffer});
            shapes.insert({device->id(), tensor.get_legacy_shape()});
        }
        return Tensor{
            MultiDeviceStorage{strategy, device_buffers, shapes},
            tensors.at(0).get_legacy_shape(),
            tensors.at(0).get_dtype(),
            tensors.at(0).get_layout()
        };
    } else if (storage_type == StorageType::MULTI_DEVICE_HOST) {
        std::vector<OwnedBuffer> owned_buffers;
        std::vector<Shape> shapes;
        for (const auto& tensor : tensors) {
            owned_buffers.push_back(std::get<OwnedStorage>(tensor.get_storage()).buffer);
            shapes.push_back(tensor.get_legacy_shape());
        }
        return Tensor{
            MultiDeviceHostStorage{strategy, owned_buffers, shapes},
            tensors.at(0).get_legacy_shape(),
            tensors.at(0).get_dtype(),
            tensors.at(0).get_layout()
        };
    } else {
        TT_THROW("Invalid storage type for multi-device tensor");
    }
}

Tensor transform(const Tensor& tensor, std::function<Tensor(const Tensor&)> transform_func) {
    auto input_tensors = get_tensors_from_multi_device_storage(tensor);
    std::vector<Tensor> output_tensors(input_tensors.size());
    std::transform(input_tensors.begin(), input_tensors.end(), output_tensors.begin(),
        [&](const auto& device_tensor) { return transform_func(device_tensor); });
    return create_multi_device_tensor(output_tensors, tensor.storage_type(), get_distributed_tensor_config_from_tensor(tensor));
}

void apply(const Tensor& tensor, std::function<void(const Tensor&)> callable) {
    auto input_tensors = get_tensors_from_multi_device_storage(tensor);
    for (const auto& device_tensor : input_tensors) {
        callable(device_tensor);
    }
}


std::vector<Device*> get_devices(const Tensor& tensor) {
    std::vector<Device*> devices;
    if (tensor.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE) {
        const auto& tensor_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        for (const auto& device_buf_pair : tensor_storage.buffers) {
            devices.push_back(device_buf_pair.second->device());
        }
        return devices;
    } else {
        TT_THROW("Tensor is not a multi-device tensor");
    }
}

uint32_t num_buffers_in_tensor(const Tensor& tensor) {
    if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage())) {
        auto device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        return device_storage.num_buffers();
    } else if (std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
        auto host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        return host_storage.num_buffers();
    } else if (std::holds_alternative<DeviceStorage>(tensor.get_storage()) || std::holds_alternative<OwnedStorage>(tensor.get_storage()) || std::holds_alternative<BorrowedStorage>(tensor.get_storage())) {
        return 1;
    } else {
        TT_FATAL(false, "get_shard_for_device only supports multi-device or device tensors");
    }
}

Tensor get_shard_for_device(const Tensor& tensor, Device* target_device) {
    if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage())) {
        auto device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        auto shard_shape = device_storage.get_tensor_shape_for_device(target_device);
        auto shard_buffer = device_storage.get_buffer_for_device(target_device);
        return Tensor{DeviceStorage{shard_buffer}, shard_shape, tensor.get_dtype(), tensor.get_layout()};
    } else if (std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
        auto host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        auto shard_shape = host_storage.get_tensor_shape_for_device(target_device);
        auto shard_buffer = host_storage.get_buffer_for_device(target_device);
        return Tensor{OwnedStorage{shard_buffer}, shard_shape, tensor.get_dtype(), tensor.get_layout()};
    } else if (std::holds_alternative<DeviceStorage>(tensor.get_storage()) || std::holds_alternative<OwnedStorage>(tensor.get_storage()) || std::holds_alternative<BorrowedStorage>(tensor.get_storage())) {
        return tensor;
    } else {
        TT_FATAL(false, "get_shard_for_device only supports multi-device or device tensors");
    }
}

void insert_buffer_and_shape_for_device(Device* target_device, const Tensor& shard, Tensor& tensor_to_modify) {
    if (std::holds_alternative<MultiDeviceHostStorage>(tensor_to_modify.tensor_attributes->storage)) {
        std::get<MultiDeviceHostStorage>(tensor_to_modify.tensor_attributes->storage).insert_buffer_and_shape_for_device(target_device, std::get<OwnedStorage>(shard.get_storage()).get_buffer(), shard.get_legacy_shape());
    } else if (std::holds_alternative<MultiDeviceStorage>(tensor_to_modify.tensor_attributes->storage)) {
        std::get<MultiDeviceStorage>(tensor_to_modify.tensor_attributes->storage).insert_buffer_and_shape_for_device(target_device, std::get<DeviceStorage>(shard.get_storage()).get_buffer(), shard.get_legacy_shape());
    } else if (std::holds_alternative<OwnedStorage>(tensor_to_modify.tensor_attributes->storage)) {
        std::get<OwnedStorage>(tensor_to_modify.tensor_attributes->storage).insert_buffer(std::get<OwnedStorage>(shard.get_storage()).get_buffer());
    } else if (std::holds_alternative<DeviceStorage>(tensor_to_modify.tensor_attributes->storage)) {
        std::get<DeviceStorage>(tensor_to_modify.tensor_attributes->storage).insert_buffer(std::get<DeviceStorage>(shard.get_storage()).get_buffer());
    } else {
        TT_FATAL(false, "Unsupported storage in insert_buffer_and_shape_for_device");
    }
}

Tensor copy_borrowed_tensor_in_async_mode(Device* worker, const Tensor& tensor) {
    // When using async mode, tensors with borrowed storage cannot be passed to workers.
    // They need to be copied to owned storage before being passed to the worker.
    ZoneScopedN("ConvertBorrowedToOwned");
    if (worker->get_worker_mode() == WorkExecutorMode::ASYNCHRONOUS and tensor.storage_type() == StorageType::BORROWED) {
        ZoneScopedN("CopyBorrowedStorage");
        auto borrowed_buffer = std::get<BorrowedStorage>(tensor.get_storage()).buffer;
        Tensor owned_tensor;
        std::visit([&owned_tensor, &tensor] (auto&& buffer) {
            using BorrowedStorageType = std::vector<std::decay_t<decltype(*(buffer.begin()))>>;
            auto owned_buf = owned_buffer::create(BorrowedStorageType(buffer.begin(), buffer.end()));
            owned_tensor = Tensor(OwnedStorage{owned_buf}, tensor.get_shape(), tensor.get_dtype(), tensor.get_layout());
        }, borrowed_buffer);
        return owned_tensor;
    }
    return tensor;
}

}

}
