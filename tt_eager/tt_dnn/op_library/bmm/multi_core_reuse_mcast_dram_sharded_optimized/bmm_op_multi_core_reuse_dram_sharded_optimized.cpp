// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include <algorithm>
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::constants;
using namespace tt;

namespace reuse_dram_sharded_optimized_helpers {
using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

void get_dram_reader_core_coords(tt_metal::Device *device, CoreRangeSet& all_cores, std::vector<CoreCoord>& all_cores_ordered) {
    uint32_t eth_coord_y_phy = 6;
    uint32_t adj_core_y_left_phy = 1;
    uint32_t adj_core_y_right_phy = 6;
    // get all the logical coord
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    std::vector<CoreCoord> all_worker_cores_log;
    for (int i=0; i<num_cores_x; ++i) {
        for (int j=0; j<num_cores_y; ++j) {
            all_worker_cores_log.push_back(CoreCoord(i,j));
        }
    }

    std::vector<uint32_t> all_worker_cores_phy;
    uint32_t max_worker_y_phy = 0;
    uint32_t min_worker_y_phy = 10000;
    for (int i=0; i<num_cores_y; ++i) {
        auto core_phy = device->worker_core_from_logical_core(CoreCoord(0,i));
        all_worker_cores_phy.push_back(core_phy.y);
        if (core_phy.y > max_worker_y_phy) {
            max_worker_y_phy = core_phy.y;
        }
        if (core_phy.y < min_worker_y_phy) {
            min_worker_y_phy = core_phy.y;
        }
    }

    std::string filename = "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml";
    YAML::Node config = YAML::LoadFile(filename);
    std::vector<CoreCoord> dram_coord_phy;
    if (config["dram_preferred_worker_endpoint"] && config["dram_preferred_worker_endpoint"].IsSequence()) {
        for (const auto& endpoint : config["dram_preferred_worker_endpoint"]) {
            std::string ep_str = endpoint.as<std::string>();
            std::istringstream iss(ep_str);
            int x, y;
            char dash; // to consume the '-' character
            if (iss >> x >> dash >> y) {
                dram_coord_phy.push_back(CoreCoord(x, y));
            }
        }
    }

    auto num_banks = dram_coord_phy.size();
    std::vector<CoreCoord> adj_core_phy;
    for (int i=0; i<num_banks; ++i) {
        auto dram_core = dram_coord_phy[i];
        uint32_t adj_core_x = dram_core.x + 1;
        uint32_t adj_core_y = dram_core.y;
        adj_core_phy.push_back(CoreCoord(adj_core_x, adj_core_y));
    }

    std::vector<CoreCoord> adj_core_phy_g1;
    std::vector<size_t> adj_core_phy_y_g1;
    std::vector<CoreCoord> adj_core_phy_g2;
    std::vector<size_t> adj_core_phy_y_g2;
    for (auto core: adj_core_phy) {
        if (core.x == adj_core_y_left_phy) {
            adj_core_phy_g1.push_back(core);
        } else if (core.x == adj_core_y_right_phy) {
            adj_core_phy_g2.push_back(core);
        }
    }
    std::vector<int> indices_g1(adj_core_phy_g1.size());
    std::vector<int> indices_g2(adj_core_phy_g2.size());
    std::iota(indices_g1.begin(), indices_g1.end(), 0);
    std::iota(indices_g2.begin(), indices_g2.end(), 0);
    std::sort(indices_g1.begin(), indices_g1.end(), [&adj_core_phy_g1](int i1, int i2) {
        return adj_core_phy_g1[i1].y < adj_core_phy_g1[i2].y;
    });
    std::sort(indices_g2.begin(), indices_g2.end(), [&adj_core_phy_g2](int i1, int i2) {
        return adj_core_phy_g2[i1].y < adj_core_phy_g2[i2].y;
    });
    std::rotate(indices_g1.begin(), indices_g1.end() - 1, indices_g1.end());
    std::rotate(indices_g2.begin(), indices_g2.end() - 1, indices_g2.end());

    std::vector<int> indices_g1_realloc(adj_core_phy_g1.size());
    std::vector<int> indices_g2_realloc(adj_core_phy_g2.size());
    for (int new_index = 0; new_index < indices_g1.size(); ++new_index) {
        indices_g1_realloc[indices_g1[new_index]] = new_index;
    }
    for (int new_index = 0; new_index < indices_g2.size(); ++new_index) {
        indices_g2_realloc[indices_g2[new_index]] = new_index;
    }

    std::sort(adj_core_phy_g1.begin(), adj_core_phy_g1.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    std::sort(adj_core_phy_g2.begin(), adj_core_phy_g2.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    std::rotate(adj_core_phy_g1.begin(), adj_core_phy_g1.end() - 1, adj_core_phy_g1.end());
    std::rotate(adj_core_phy_g2.begin(), adj_core_phy_g2.end() - 1, adj_core_phy_g2.end());

    for (auto core: adj_core_phy_g1) {
        adj_core_phy_y_g1.push_back(core.y);
    }
    for (auto core: adj_core_phy_g2) {
        adj_core_phy_y_g2.push_back(core.y);
    }

    std::vector<uint32_t> harvested_rows;
    std::vector<uint32_t> all_phy_coord_y = {1,2,3,4,5,7,8,9,10,11};
    uint32_t max_bank_id = 11;
    // auto min_core_storage = storage_core_range.start;
    // auto max_core_storage = storage_core_range.end;
    // std::vector<uint32_t> storage_core_y_phy;
    // for (int i=min_core_storage.y; i<=max_core_storage.y; ++i) {
    //     auto core_phy = device->worker_core_from_logical_core(CoreCoord(0,i));
    //     harvested_rows.push_back(core_phy.y);
    // }
    for (int i=0; i<all_phy_coord_y.size(); ++i) {
        auto y = all_phy_coord_y[i];

        if (std::find(all_worker_cores_phy.begin(), all_worker_cores_phy.end(), y) == all_worker_cores_phy.end()) {
            harvested_rows.push_back(y);
        }
    }

    uint32_t x_step = 3;
    for (int i=0; i<adj_core_phy_g1.size(); ++i) {
        auto y = adj_core_phy_g1[i].y;

        if (std::find(harvested_rows.begin(), harvested_rows.end(), y) != harvested_rows.end() or
            std::count(adj_core_phy_y_g1.begin(), adj_core_phy_y_g1.end(), y) >= 2) {

            if (y >= max_bank_id) {
                for (int j=max_worker_y_phy; j >= min_worker_y_phy; j--) {
                    auto temp_y = j;

                    if (std::find(harvested_rows.begin(), harvested_rows.end(), temp_y) == harvested_rows.end() and
                        std::count(adj_core_phy_y_g1.begin(), adj_core_phy_y_g1.end(), temp_y) == 0) {

                        adj_core_phy_g1[i].y = temp_y;
                        adj_core_phy_g1[i].x += x_step;
                        x_step --;
                        break;
                    }
                }
            } else {
                for (int j=min_worker_y_phy; j <= max_worker_y_phy; j++) {
                    auto temp_y = j;
                    if (std::find(harvested_rows.begin(), harvested_rows.end(), temp_y) == harvested_rows.end() and
                        std::count(adj_core_phy_y_g1.begin(), adj_core_phy_y_g1.end(), temp_y) == 0) {

                        adj_core_phy_g1[i].y = temp_y;
                        adj_core_phy_g1[i].x += x_step;
                        x_step --;
                        break;
                    }
                }
            }
        }
    }

    x_step = 3;
    for (int i=0; i<adj_core_phy_g2.size(); ++i) {
        auto y = adj_core_phy_g2[i].y;

        if (std::find(harvested_rows.begin(), harvested_rows.end(), y) != harvested_rows.end() or
            std::count(adj_core_phy_y_g2.begin(), adj_core_phy_y_g2.end(), y) >= 2) {

            if (y >= max_bank_id) {
                for (int j=max_worker_y_phy; j >= min_worker_y_phy; j--) {
                    auto temp_y = j;

                    if (std::find(harvested_rows.begin(), harvested_rows.end(), temp_y) == harvested_rows.end() and
                        std::count(adj_core_phy_y_g2.begin(), adj_core_phy_y_g2.end(), temp_y) == 0 and temp_y != eth_coord_y_phy) {

                        adj_core_phy_g2[i].y = temp_y;
                        adj_core_phy_g2[i].x += x_step;
                        x_step --;
                        break;
                    }
                }
            } else {
                for (int j=min_worker_y_phy; j <= max_worker_y_phy; j++) {
                    auto temp_y = j;
                    if (std::find(harvested_rows.begin(), harvested_rows.end(), temp_y) == harvested_rows.end() and
                        std::count(adj_core_phy_y_g2.begin(), adj_core_phy_y_g2.end(), temp_y) == 0 and temp_y != eth_coord_y_phy) {

                        adj_core_phy_g2[i].y = temp_y;
                        adj_core_phy_g2[i].x += x_step;
                        x_step --;
                        break;
                    }
                }
            }
        }
    }

    std::vector<CoreCoord> adj_core_phy_new;
    for (int i=0; i<indices_g1_realloc.size(); ++i) {
        adj_core_phy_new.push_back(adj_core_phy_g1[indices_g1_realloc[i]]);
    }
    for (int i=0; i<indices_g2_realloc.size(); ++i) {
        adj_core_phy_new.push_back(adj_core_phy_g2[indices_g2_realloc[i]]);
    }

    std::vector<CoreCoord> all_cores_log;
    for (int i=0; i < adj_core_phy_new.size(); ++i) {
        auto core_phy = adj_core_phy_new[i];

        for (int j=0; j < all_worker_cores_log.size(); ++j) {
            auto core_phy_ = device->worker_core_from_logical_core(all_worker_cores_log[j]);
            if (core_phy == core_phy_) {
                all_cores_log.push_back(all_worker_cores_log[j]);
            }
        }
    }

    std::set<CoreRange> all_cores_set;
    for (int i=0; i<num_banks; ++i) {
        all_cores_set.insert(CoreRange(all_cores_log[i]));
    }

    all_cores = CoreRangeSet(all_cores_set);
    all_cores_ordered = all_cores_log;
}

void get_max_page_size_and_num_pages(uint32_t num_tiles, uint32_t tile_size, uint32_t &page_size, uint32_t &num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * tile_size;

    page_size = (8192 / tile_size) * tile_size;
    while (total_size % page_size != 0 && page_size >= tile_size) {
        page_size -= tile_size;
    }
    num_pages = total_size / page_size;
}

void move_common_entries(std::vector<CoreCoord>& v1, std::vector<CoreCoord>& v2, std::vector<CoreCoord>& commons) {

    for (const CoreCoord& item : v2) {
        if (std::find(v1.begin(), v1.end(), item) != v1.end()) {
            commons.push_back(item);
        }
    }

    for (const CoreCoord& item : commons) {
        v2.erase(std::remove(v2.begin(), v2.end(), item), v2.end());
    }
}

uint32_t get_num_groups_per_block(uint32_t in0_block_w, uint32_t max_group_size) {
    uint32_t best_group_size = max_group_size;

    while (in0_block_w % best_group_size != 0 && best_group_size > 1) {
        best_group_size--;
    }

    uint32_t num_groups = in0_block_w / best_group_size;
    return num_groups;
}

operation::ProgramWithCallbacks create_program_dram_sharded(
    tt_metal::Device *device,
    CoreRangeSet all_storage_cores,
    MathFidelity math_fidelity, bool fp32_dest_acc_en, bool math_approx_mode, bool packer_l1_acc,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h_storage, uint32_t out_subblock_w_storage,
    uint32_t per_core_M, uint32_t per_core_K, uint32_t per_core_N_storage,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer, tt_metal::Buffer* in1_buffer, tt_metal::Buffer* bias_buffer, tt_metal::Buffer* out_buffer,
    tt::DataFormat in0_data_format, tt::DataFormat in1_data_format, tt::DataFormat bias_data_format, tt::DataFormat output_data_format,
    bool untilize_out
) {

    log_info("math_fidelity: {}", math_fidelity);
    log_info("fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_info("math_approx_mode: {}", math_approx_mode);
    log_info("packer_l1_acc: {}", packer_l1_acc);
    log_info("M: {}, K: {}, N: {}", M, K, N);
    log_info("per_core_M: {}, per_core_N_storage: {}", per_core_M, per_core_N_storage);

    tt_metal::Program program{};

    // get the dram readers
    CoreRangeSet all_worker_cores = CoreRangeSet{{}};
    std::vector<CoreCoord> all_worker_cores_ordered;
    get_dram_reader_core_coords(device, all_worker_cores, all_worker_cores_ordered);

    // dram banks
    uint32_t num_dram_banks = all_worker_cores_ordered.size();
    for (auto core : corerange_to_cores(all_worker_cores)) {
        log_info("all_worker_cores_log: {}", core);
    }
    for (auto core : all_worker_cores_ordered) {
        log_info("all_worker_cores_ordered: {}", core);
    }

    uint32_t per_core_N = (N + num_dram_banks - 1) / num_dram_banks;

    log_info("per_core_M: {}, per_core_N: {}", per_core_M, per_core_N);


    auto subblock_hw = bmm_op_utils::get_matmul_subblock_params(per_core_M, per_core_N, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    log_info("out_subblock_h: {}, out_subblock_w: {}", out_subblock_h, out_subblock_w);

    uint32_t num_transaction_groups_per_block = get_num_groups_per_block(in0_block_w, 16); // 16 tiles maxmimum in block height

    log_info("num_transaction_groups_per_block: {}", num_transaction_groups_per_block);

    uint32_t num_blocks = K / in0_block_w;
    //Only enable packer l1 accumulation when there are spills, otherwise
    //unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);
    interm0_data_format = tt::DataFormat::Float16_b;

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2; // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 2; // double buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles; // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t out_reshard_block_tiles = per_core_M * per_core_N_storage;
    uint32_t out_reshard_CB_tiles = out_reshard_block_tiles; // No double buffer
    uint32_t out_reshard_CB_size = out_reshard_CB_tiles * output_single_tile_size;

    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / TILE_WIDTH;
    uint32_t in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / TILE_HEIGHT;
    uint32_t in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N;
    uint32_t in3_CB_tiles = in3_block_tiles; // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    // get the max page size based on num tiles
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    uint32_t num_worker_cores = num_dram_banks;
    uint32_t num_mcast_cores = num_worker_cores;

    // move conflict coord from mcast receiver to mcast sender
    std::vector<CoreCoord> all_storage_cores_vec = corerange_to_cores(all_storage_cores);
    std::vector<CoreCoord> all_worker_cores_vec = corerange_to_cores(all_worker_cores);
    std::vector<CoreCoord> storage_worker_common;
    move_common_entries(all_storage_cores_vec, all_worker_cores_vec, storage_worker_common);

    std::vector<CoreRange> all_storage_cores_range;
    all_storage_cores_range.reserve(all_storage_cores_vec.size());
    std::transform(all_storage_cores_vec.begin(), all_storage_cores_vec.end(),
                   std::back_inserter(all_storage_cores_range),
                   [](const CoreCoord& coord) { return CoreRange(coord); });

    std::vector<CoreRange> all_worker_cores_range;
    all_worker_cores_range.reserve(all_worker_cores_vec.size());
    std::transform(all_worker_cores_vec.begin(), all_worker_cores_vec.end(),
                   std::back_inserter(all_worker_cores_range),
                   [](const CoreCoord& coord) { return CoreRange(coord); });


    std::set<CoreRange> all_storage_cores_set(all_storage_cores_range.begin(), all_storage_cores_range.end());
    std::set<CoreRange> all_worker_cores_set(all_worker_cores_range.begin(), all_worker_cores_range.end());
    CoreRangeSet mcast_senders = CoreRangeSet(all_storage_cores_set);
    CoreRangeSet mcast_receivers = CoreRangeSet(all_worker_cores_set);

    for (auto core : corerange_to_cores(mcast_senders)) {
        log_info("mcast_senders: {}", core);
    }
    for (auto core : corerange_to_cores(mcast_receivers)) {
        log_info("mcast_receivers: {}", core);
    }

    // all cores
    std::set<CoreRange> all_cores_set;
    all_cores_set.insert(mcast_senders.ranges().begin(), mcast_senders.ranges().end());
    all_cores_set.insert(mcast_receivers.ranges().begin(), mcast_receivers.ranges().end());
    CoreRangeSet all_cores = CoreRangeSet(all_cores_set);

    for (auto core : corerange_to_cores(all_cores)) {
        log_info("all_cores: {}", core);
    }

    // Mcast args
    auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_sender_valid_semaphore = tt_metal::CreateSemaphore(program, all_cores, VALID);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    CoreCoord top_left_core = {(std::size_t) start_core_x, (std::size_t) start_core_y};
    CoreCoord bottom_right_core = {(std::size_t) start_core_x + compute_with_storage_grid_size.x - 1, (std::size_t) start_core_y + compute_with_storage_grid_size.y - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = false;
    bool in1_is_dram = true;
    bool in3_is_dram = true;

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = detail::GetPreferredNOCForDRAMRead(device->arch());

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)  in0_block_num_tiles, // in0_block_num_tiles
        (std::uint32_t)  in0_block_num_tiles * in0_single_tile_size, // in0_block_size_bytes
        // in0 mcast args
        (std::uint32_t)  in0_mcast_sender_semaphore,
        (std::uint32_t)  in0_mcast_receiver_semaphore,
        (std::uint32_t)  num_worker_cores, // in0_mcast_num_dests
        (std::uint32_t)  num_mcast_cores, // in0_mcast_num_cores
        // block
        (std::uint32_t)  num_blocks,
        // mcast noc coords
        (std::uint32_t)  start_core_noc.x,
        (std::uint32_t)  start_core_noc.y,
        (std::uint32_t)  end_core_noc.x,
        (std::uint32_t)  end_core_noc.y,
        // semahpre valid
        (std::uint32_t)  in0_mcast_sender_valid_semaphore
    };

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // (std::uint32_t)  in1_buffer->address(),
        (std::uint32_t)  in1_buffer_page_size,
        (std::uint32_t)  in1_buffer_num_pages / num_transaction_groups_per_block,
        // in1 block args
        (std::uint32_t)  per_core_N, // in1_block_w
        (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)  num_blocks, // num_blocks
        (std::uint32_t)  out_block_tiles, // out_block_num_tiles
        (std::uint32_t)  per_core_N * output_single_tile_size, // out_tensor_stride_w_bytes
        (std::uint32_t)  per_core_N_storage * output_single_tile_size, // out_reshard_tensor_stride_w_bytes
        (std::uint32_t)  per_core_M,
        (std::uint32_t)  num_transaction_groups_per_block
    };
    if (bias_buffer != nullptr) {
        // in1_sender_writer_compile_time_args.push_back(bias_buffer->address());
        in1_sender_writer_compile_time_args.push_back(bias_buffer_page_size);
        in1_sender_writer_compile_time_args.push_back(bias_buffer_num_pages);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)  1);
    }
    std::vector<uint32_t> in0_receiver_compile_time_args = {
        // in0 block args
        (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)  num_blocks, // num_blocks
        // in0 mcast args
        (std::uint32_t)  in0_mcast_sender_semaphore,
        (std::uint32_t)  in0_mcast_receiver_semaphore,
        // batch args
        (std::uint32_t)  B, // batch
    };

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines.merge(eltwise_unary_op_utils::get_defines(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    if (num_transaction_groups_per_block > 1) {
        mm_kernel_defines["MULTIPLE_GROUPS_PER_BLOCK"] = "1";
        mm_kernel_in1_sender_writer_defines["MULTIPLE_GROUPS_PER_BLOCK"] = "1";
    }

    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp",
        mcast_senders,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc, .compile_args = in0_sender_compile_time_args});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp",
        all_worker_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc, .compile_args = in1_sender_writer_compile_time_args, .defines = mm_kernel_in1_sender_writer_defines});

    KernelHandle mm_kernel_in0_receiver_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_dram_sharded.cpp",
        mcast_receivers,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc, .compile_args = in0_receiver_compile_time_args});

    // Compute kernel compile time args
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w / num_transaction_groups_per_block, // in0_block_w
        in0_num_subblocks, // in0_num_subblocks
        in0_block_num_tiles, // in0_block_num_tiles
        in0_subblock_num_tiles, // in0_subblock_num_tiles

        in1_num_subblocks, // in1_num_subblocks
        in1_block_num_tiles, // in1_block_num_tiles
        in1_per_core_w, // in1_per_core_w

        num_blocks, // num_blocks

        out_subblock_h, // out_subblock_h
        out_subblock_w, // out_subblock_w
        out_subblock_num_tiles, // out_subblock_num_tiles
        B, // batch
        out_block_tiles, // out_block_num_tiles

        untilize_out, // untilize_out
        num_transaction_groups_per_block,
        in1_block_num_tiles / num_transaction_groups_per_block
    };

    // Create compute kernel
    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/bmm/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_worker_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode, .compile_args = compute_kernel_args, .defines = mm_kernel_defines}
    );

    log_info(LogOp, "in1_single_tile_size: {}", in1_single_tile_size);

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
		.set_page_size(src0_cb_index, in0_single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    // auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_worker_cores, src0_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", src0_cb_index, in0_single_tile_size, in0_CB_size / in0_single_tile_size, in0_CB_size);


    uint32_t src1_cb_index = 1;
    std::map<uint8_t, tt::DataFormat> in1_cb_data_format_spec {
        {src1_cb_index, in1_data_format}
    };
    for (uint32_t i = 0; i < num_transaction_groups_per_block - 1; ++i) {
        in1_cb_data_format_spec.insert({src1_cb_index + i + 3, in1_data_format});
    }
    tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, in1_cb_data_format_spec)
        .set_page_size(1, in1_single_tile_size);

    for (uint32_t i = 0; i < num_transaction_groups_per_block - 1; ++i) {
        src1_cb_config = src1_cb_config.set_page_size(src1_cb_index + i + 3, in1_single_tile_size);
    }

    // uint32_t src1_cb_index = 1;
    // tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
	// 	.set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    // auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_worker_cores, src1_cb_config);
    // log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", src1_cb_index, in1_single_tile_size, in1_CB_size / in1_single_tile_size, in1_CB_size);

    uint32_t src2_cb_index = 2;
    tt_metal::CircularBufferConfig src2_cb_config = tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
        .set_page_size(src2_cb_index, in0_single_tile_size).set_globally_allocated_address(*in0_buffer);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);
    // auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_storage_cores, src2_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", src2_cb_index, in0_single_tile_size, in2_CB_size / in0_single_tile_size, in2_CB_size);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    tt_metal::CircularBufferConfig interm0_cb_config = tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // output
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
            {output_cb_index, output_data_format},
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, output_single_tile_size);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec {
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
            .set_page_size(interm0_cb_index, interm0_single_tile_size);

        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_cores, interm0_cb_config);
        // auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_worker_cores, interm0_cb_config);
        log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", interm0_cb_index, interm0_single_tile_size, interm0_CB_size / interm0_single_tile_size, interm0_CB_size);
    } else {
        log_info(LogOp, "inplace interm and outout cb");
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
            {output_cb_index, output_data_format},
            {interm0_cb_index, interm0_data_format}
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
            .set_page_size(output_cb_index, output_single_tile_size)
            .set_page_size(interm0_cb_index, interm0_single_tile_size);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    // auto cb_output = tt_metal::CreateCircularBuffer(program, all_worker_cores, output_cb_config);
    log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", output_cb_index, output_single_tile_size, out_CB_size / output_single_tile_size, out_CB_size);


    // resharded output
    uint32_t output_reshard_cb_index = 17;
    std::map<uint8_t, tt::DataFormat> output_reshard_cb_data_format_spec {
        {output_reshard_cb_index, output_data_format},
    };
    tt_metal::CircularBufferConfig output_reshard_cb_config = tt_metal::CircularBufferConfig(out_reshard_CB_size, output_reshard_cb_data_format_spec)
        .set_page_size(output_reshard_cb_index, output_single_tile_size);
    output_reshard_cb_config = output_reshard_cb_config.set_globally_allocated_address(*out_buffer);
    auto cb_output_reshard = tt_metal::CreateCircularBuffer(program, all_cores, output_reshard_cb_config);
    // auto cb_output_reshard = tt_metal::CreateCircularBuffer(program, all_storage_cores, output_reshard_cb_config);


    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = 3;
        tt_metal::CircularBufferConfig cb_src3_config = tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
		    .set_page_size(src3_cb_index, bias_single_tile_size);
        auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);
        log_debug(LogOp, "CB {} :: PS = {}, NP = {}, TOTAL = {}", src3_cb_index, bias_single_tile_size, in3_CB_size / bias_single_tile_size, in3_CB_size);
    }

    // Parameters for last row, col, or block
    uint32_t last_block_h = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_block_w = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_block_num_nonzero_subblocks_h = (last_block_h  - 1) / out_subblock_h + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_block_w  - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_h = last_block_h % out_subblock_h == 0 ? out_subblock_h : last_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w = last_block_w % out_subblock_w == 0 ? out_subblock_w : last_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip = output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =  (out_subblock_w * out_subblock_h) * (per_core_N / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip = (per_core_M / out_subblock_h - last_block_num_nonzero_subblocks_h) * (per_core_N * out_subblock_h);

    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;

    std::vector<uint32_t> in0_mcast_sender_noc_x;
    std::vector<uint32_t> in0_mcast_sender_noc_y;
    std::vector<CoreCoord> mcast_senders_coords = corerange_to_cores(mcast_senders);
    std::sort(mcast_senders_coords.begin(), mcast_senders_coords.end(),
        [](const CoreCoord& a, const CoreCoord& b) {
            if (a.y != b.y) {
                return a.y < b.y;
            }
            return a.x < b.x;
        });
    for(auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_x.push_back((std::uint32_t) device->worker_core_from_logical_core(core).x);
    }
    for(auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_y.push_back((std::uint32_t) device->worker_core_from_logical_core(core).y);
    }

    uint32_t sender_id = 0;
    for(auto core : mcast_senders_coords) {
        std::vector<uint32_t> mm_in0_sender_args;

        bool is_worker_core;
        if (find(storage_worker_common.begin(), storage_worker_common.end(), core) != storage_worker_common.end()) {
            is_worker_core = true;
        } else {
            is_worker_core = false;
        }

        mm_in0_sender_args.push_back((std::uint32_t) is_worker_core);
        mm_in0_sender_args.push_back((std::uint32_t) sender_id);
        mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);

        sender_id ++;
    }

    std::vector<CoreCoord> mcast_receiver_coords = corerange_to_cores(mcast_receivers);
    for(uint32_t i=0; i < mcast_receiver_coords.size(); ++i) {
        auto core = mcast_receiver_coords[i];

        // in0 receivers rt args
        std::vector<uint32_t> mm_in0_receiver_args;
        mm_in0_receiver_args.insert(mm_in0_receiver_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_receiver_args.insert(mm_in0_receiver_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_receiver_id, core, mm_in0_receiver_args);
        reader_kernel_ids.push_back(mm_kernel_in0_receiver_id);
    }

    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;
    uint32_t curr_storage_core_idx = 0;
    uint32_t per_core_N_storage_curr_stride = 0;

    uint32_t worker_core_stride = 0;
    uint32_t storage_core_stride = 0;
    uint32_t curr_worker_core = 0;
    uint32_t curr_storage_core = 0;

    for(uint32_t i=0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];

        // in1 reader rt args
        std::vector<uint32_t> mm_in1_sender_writer_args;
        mm_in1_sender_writer_args.push_back(in1_buffer->address());
        if (bias_buffer != nullptr) {
            mm_in1_sender_writer_args.push_back(bias_buffer->address());
        } else {
            mm_in1_sender_writer_args.push_back(0);
        }

        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < i; ++j) {
            auto core_prev = all_worker_cores_ordered[j];

            if (core_prev.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) { // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }
        mm_in1_sender_writer_args.push_back((std::uint32_t) bank_id);
        mm_in1_sender_writer_args.push_back((std::uint32_t) vc);

        bank_id = (bank_id + 1) % num_dram_banks;

        if (per_core_N < per_core_N_storage) {

            if (curr_storage_core_idx < all_storage_cores_vec.size()) {
                uint32_t remaining_per_core_N_storage = (per_core_N_storage - per_core_N_storage_curr_stride);
                uint32_t per_core_N_reshard_1 = (remaining_per_core_N_storage > per_core_N) ? per_core_N : remaining_per_core_N_storage;
                uint32_t per_core_N_reshard_2 = per_core_N - per_core_N_reshard_1;

                if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < all_storage_cores_vec.size()) {
                    mm_in1_sender_writer_args.push_back(2);
                    // mm_in1_sender_writer_args.push_back(true); // split output tensor to two shards
                } else {
                    mm_in1_sender_writer_args.push_back(1);
                    // mm_in1_sender_writer_args.push_back(false);
                }

                mm_in1_sender_writer_args.push_back(per_core_N_storage_curr_stride * output_single_tile_size); // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(per_core_N_reshard_1 * output_single_tile_size); // per_core_N_reshard_bytes_1
                mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_x[curr_storage_core_idx]); // in0_mcast_sender_noc_x
                mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_y[curr_storage_core_idx]); // in0_mcast_sender_noc_y


                if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < all_storage_cores_vec.size()) {
                    mm_in1_sender_writer_args.push_back(per_core_N_reshard_2 * output_single_tile_size); // per_core_N_reshard_bytes_2
                    mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_x[curr_storage_core_idx + 1]); // in0_mcast_sender_noc_x
                    mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_y[curr_storage_core_idx + 1]); // in0_mcast_sender_noc_y
                }

                curr_storage_core_idx += (per_core_N_storage_curr_stride + per_core_N) / per_core_N_storage;
                per_core_N_storage_curr_stride = (per_core_N_storage_curr_stride + per_core_N) % per_core_N_storage;
            }
        } else {
            // uint32_t num_iter = (per_core_N + per_core_N_storage - 1) / per_core_N_storage;
            uint32_t num_iter = 0;
            // mm_in1_sender_writer_args.push_back(num_iter);

            if (curr_storage_core < all_storage_cores_vec.size()) {
                num_iter ++;

                log_info("curr worker core: {}, send back to storage core: {}, coord: {}", curr_worker_core, curr_storage_core, mcast_senders_coords[curr_storage_core]);

                worker_core_stride = per_core_N_storage - storage_core_stride;

                mm_in1_sender_writer_args.push_back(storage_core_stride * output_single_tile_size); // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(worker_core_stride * output_single_tile_size); // per_core_N_reshard
                mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_x[curr_storage_core]); // in0_mcast_sender_noc_x
                mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_y[curr_storage_core]); // in0_mcast_sender_noc_y

                curr_storage_core += (storage_core_stride + worker_core_stride) / per_core_N_storage;
                storage_core_stride = (storage_core_stride + worker_core_stride) % per_core_N_storage;

                if (worker_core_stride >= per_core_N) {
                    curr_worker_core += 1;
                }

                while(curr_worker_core <= i and curr_storage_core < all_storage_cores_vec.size()) {
                    num_iter ++;

                    log_info("curr worker core: {}, send back to storage core: {}, coord: {}", curr_worker_core, curr_storage_core, mcast_senders_coords[curr_storage_core]);

                    uint32_t stride = worker_core_stride + per_core_N_storage;
                    if (stride >= per_core_N) {
                        stride = per_core_N;
                        curr_worker_core += 1;
                    }

                    mm_in1_sender_writer_args.push_back((stride - worker_core_stride) * output_single_tile_size); // per_core_N_reshard
                    mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_x[curr_storage_core]); // in0_mcast_sender_noc_x
                    mm_in1_sender_writer_args.push_back(in0_mcast_sender_noc_y[curr_storage_core]); // in0_mcast_sender_noc_y

                    storage_core_stride = (stride - worker_core_stride) % per_core_N_storage;
                    curr_storage_core += (stride - worker_core_stride) / per_core_N_storage;
                    worker_core_stride = stride;

                }
            }

            mm_in1_sender_writer_args.insert(mm_in1_sender_writer_args.begin() + 4, num_iter);
        }

        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);
        writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
    }

    auto override_runtime_arguments_callback = [
            writer_kernel_ids,
            all_worker_cores_ordered,
            cb_src2,
            cb_output
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        TT_FATAL(input_tensors.size() + optional_input_tensors.size() == 3);
        TT_FATAL(output_tensors.size() == 1);

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();
        auto bias_tensor = optional_input_tensors.at(0);

        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src2, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);

        for(uint32_t i=0; i < all_worker_cores_ordered.size(); ++i) {
            auto core = all_worker_cores_ordered[i];
            auto writer_kernel_id = writer_kernel_ids[i];
            auto &writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_runtime_args[0] = src_buffer_b->address();
            if (bias_tensor.has_value()) {
                writer_runtime_args[1] = bias_tensor.value().buffer()->address();
            } else {
                writer_runtime_args[1] = 0;
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}
}

namespace tt {

namespace tt_metal {


operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized_(const Tensor &a, const Tensor &b, const std::optional<const Tensor> bias, Tensor& output, bool bcast_batch, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_K, uint32_t per_core_N, bool fuse_batch, std::optional<UnaryWithParam> fused_activation, bool untilize_out) {

    const auto& ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype()); // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype()); // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype()); // output

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b; // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE);
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.get_dtype());
    }

    tt_metal::Device *device = a.device();

    TT_FATAL(a.shard_spec().has_value() && output.shard_spec().has_value());
    CoreRangeSet all_cores_storage = a.shard_spec().value().grid;

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();
    tt_metal::Buffer *in1_buffer = b.buffer();
    if (bcast_batch)
        TT_FATAL(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_FATAL(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_FATAL(in0_buffer->size() % in0_single_tile_size == 0);
    TT_FATAL(in1_buffer->size() % in1_single_tile_size == 0);

    TT_FATAL(ashape[3] == bshape[2] && "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
    TT_FATAL(ashape[2] % TILE_HEIGHT == 0);
    TT_FATAL(ashape[3] % TILE_WIDTH == 0);
    TT_FATAL(bshape[2] % TILE_HEIGHT == 0);
    TT_FATAL(bshape[3] % TILE_WIDTH == 0);

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_FATAL(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
            packer_l1_acc = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_FATAL(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
            packer_l1_acc = compute_kernel_config.packer_l1_acc;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = ashape[0]*ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;

    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }
    TT_FATAL(Kt % in0_block_w == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    return reuse_dram_sharded_optimized_helpers::create_program_dram_sharded(
        device,
        all_cores_storage,
        math_fidelity, fp32_dest_acc_en, math_approx_mode, packer_l1_acc,
        B, Mt, Nt, Kt,
        bcast_batch,
        in0_block_w,
        out_subblock_h, out_subblock_w,
        per_core_M, per_core_K, per_core_N,
        fused_activation,
        in0_buffer, in1_buffer, bias_buffer, out_buffer,
        in0_data_format, in1_data_format, bias_data_format, output_data_format,
        untilize_out
    );
}

operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized(const Tensor& a, const Tensor& b, const std::optional<const Tensor> bias, Tensor& output_tensor, bool broadcast_batch, DeviceComputeKernelConfig compute_kernel_config, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_K, uint32_t per_core_N, bool fuse_batch, std::optional<UnaryWithParam> fused_activation, bool untilize_out) {
    return matmul_multi_core_reuse_dram_sharded_optimized_(a, b, bias, output_tensor, broadcast_batch, compute_kernel_config, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_K, per_core_N, fuse_batch, fused_activation, untilize_out);
}

}  // namespace tt_metal

}  // namespace tt
