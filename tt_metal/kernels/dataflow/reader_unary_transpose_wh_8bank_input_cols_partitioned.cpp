#include <stdint.h>
#include "dataflow_kernel_api.h"

//#include "debug_print.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    // skip args 1,2,3 for compat with reader_unary, reader_unary_8bank
    uint32_t N = get_arg_val<uint32_t>(4); // args match the order of reader_unary
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t start_id = get_arg_val<uint32_t>(8); // Start id in column major order
    uint32_t num_tiles = get_arg_val<uint32_t>(9); // number of tiles to read in column major order
    uint32_t scaler = get_arg_val<uint32_t>(10);
    if (scaler != 0) {
        union { float f; uint32_t u; } u; u.u = scaler;
        //DPRINT << "TWH ic part Scaler = " << F32(u.f) << ENDL();
        constexpr uint32_t cb_in_2 = 2;
        dataflow::cb_reserve_back(cb_in_2, 1);
        auto ptr = reinterpret_cast<uint16_t*>(dataflow::get_write_ptr(cb_in_2));
        for (int j = 0; j < 1024; j++)
            ptr[j] = uint16_t(0);

        for (int k = 0; k < 4; k++)
        for (int j = 0; j < 16; j++)
            ptr[k*256 + j] = uint16_t(u.u>>16);
        dataflow::cb_push_back(cb_in_2, 1);
    }

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const dataflow::InterleavedPow2AddrGen<true> s = {
        .bank_base_address = src_addr,


        .log_base_2_of_page_size = 11
    };
    uint32_t h = start_id % Ht;
    uint32_t w = start_id / Ht % Wt;
    uint32_t nc = start_id / HtWt;
    uint32_t i_tile = nc * HtWt + w + h * Wt;

    // this reader will read a NHW tensor in NWH order
    for (uint32_t i = 0; i < num_tiles; i++){
        uint64_t src_noc_addr = dataflow::get_noc_addr(i_tile, s);
        dataflow::cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = dataflow::get_write_ptr(cb_id_in0);
        dataflow::noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
        dataflow::noc_async_read_barrier();

        dataflow::cb_push_back(cb_id_in0, onetile);
        i_tile += Wt; // stride in H
        h += 1;
        if (h == Ht) {
            h = 0;
            i_tile += 1;
            w += 1;
            if (w == Wt) {
                w = 0;
                i_tile -= Wt; // Start of next batch
            } else {
                i_tile -= HtWt; // Start of next col
            }
        }
    }
}
