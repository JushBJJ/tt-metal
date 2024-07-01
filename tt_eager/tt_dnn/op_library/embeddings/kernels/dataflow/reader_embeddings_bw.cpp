// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

constexpr uint32_t INPUT_SIZE = 32;

constexpr bool grad_is_dram = get_compile_time_arg_val(0) == 1;
constexpr bool out_is_dram = get_compile_time_arg_val(2) == 1;

constexpr uint32_t cb_id_in0 = 0;
constexpr uint32_t cb_id_in1 = 1;
constexpr uint32_t cb_id_intermed0 = 24;
constexpr uint32_t cb_id_out0 = 16;

FORCE_INLINE uint64_t get_index_noc_address(uint32_t tile_idx, uint32_t offset = 0) {
    const std::uint32_t index_tensor_addr = get_arg_val<uint32_t>(1);
    constexpr bool index_stick_size_is_power_of_two = get_compile_time_arg_val(4) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(1) == 1;

    if constexpr (index_stick_size_is_power_of_two) {
        constexpr uint32_t index_log2_stick_size = get_compile_time_arg_val(5);
        InterleavedPow2AddrGen<index_is_dram> index = {
            .bank_base_address = index_tensor_addr, .log_base_2_of_page_size = index_log2_stick_size};
        return get_noc_addr(tile_idx, index, offset);
    } else {
        constexpr uint32_t index_page_size = get_compile_time_arg_val(3);
        InterleavedAddrGen<index_is_dram> index = {
            .bank_base_address = index_tensor_addr, .page_size = index_page_size};
        return get_noc_addr(tile_idx, index, offset);
    }
}

FORCE_INLINE uint32_t get_index(uint32_t input_l1_addr, uint32_t idx) {
    constexpr bool is_index_bfloat16 = get_compile_time_arg_val(6) == 1;
    if constexpr (is_index_bfloat16) {
        auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(input_l1_addr);
        union {
            float f;
            uint32_t u;
        } u;
        u.u = (uint32_t)input_l1_ptr[idx] << 16;
        return static_cast<uint32_t>(u.f);
    } else {
        auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(input_l1_addr);
        return input_l1_ptr[idx];
    }
}

FORCE_INLINE uint32_t process_index_chunk(uint32_t index_l1_addr, uint32_t chunk_indexes[INPUT_SIZE]) {
    uint32_t chunk_count = 0;

    for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
        uint32_t idx = get_index(index_l1_addr, i);
        uint32_t chunk_id = idx >> 5;  // equivalent to idx / 32

        bool is_new_chunk = true;
        for (uint32_t j = 0; j < chunk_count; ++j) {
            if (chunk_indexes[j] == chunk_id) {
                is_new_chunk = false;
                break;
            }
        }

        if (is_new_chunk) {
            chunk_indexes[chunk_count++] = chunk_id;
        }
    }

    return chunk_count;
}

FORCE_INLINE void generate_mask(uint32_t index_l1_addr, uint32_t chunk_id, uint32_t mask_l1_addr) {
    auto mask_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(mask_l1_addr);

    uint32_t x_min = chunk_id << 5;  // equivalent to chunk_id * 32
    uint32_t x_max = x_min + INPUT_SIZE;

    for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
        uint32_t idx = get_index(index_l1_addr, i);
        uint32_t mask = ~static_cast<uint32_t>(0);  // equivalent to numeric_limits<uint32_t>::max()
        if (idx >= x_min && idx < x_max) {
            mask = idx & (INPUT_SIZE - 1);  // equivalent to idx % INPUT_SIZE
        }
        mask_l1_ptr[i] = mask;
    }
}

void kernel_main() {
    const uint32_t grad_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t seq_tile_len = get_arg_val<uint32_t>(3);
    const uint32_t batch_size = get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_hidden = get_arg_val<uint32_t>(5);
    const uint32_t hidden_offset = get_arg_val<uint32_t>(6);
    const uint32_t tiles_per_core = get_arg_val<uint32_t>(7);

    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_id_in1);
    uint32_t index_block_size = get_tile_size(cb_id_in1) >> 5;  // we only need 32 elements
    uint32_t mask_l1_addr = get_write_ptr(cb_id_intermed0);

    uint32_t chunk_indexes[INPUT_SIZE];

    for (uint32_t b = 0; b < batch_size; ++b) {
        auto next_index_seq_addr = get_index_noc_address(b);
        uint32_t offset = 0;
        for (uint32_t s = 0; s < seq_tile_len; ++s) {
            noc_async_read(next_index_seq_addr + offset, input_l1_addr, index_block_size);
            noc_async_read_barrier();
            offset += index_block_size;

            // maps the next chunk of indexes to the corresponding output masks
            uint32_t chunk_count = process_index_chunk(input_l1_addr, chunk_indexes);
            for (uint32_t i = 0; i < chunk_count; ++i) {
                generate_mask(input_l1_addr, chunk_indexes[i], mask_l1_addr);

                for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
                    uint32_t idx = get_index(input_l1_addr, i);
                    uint32_t msk = get_index(mask_l1_addr, i);
                    DPRINT << i << ": " << idx << " -> " << msk << ENDL();
                }
            }
        }
    }
}
